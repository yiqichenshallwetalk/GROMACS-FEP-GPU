/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2018- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 *
 * \brief Implements CUDA bonded functionality
 *
 * \author Jon Vincent <jvincent@nvidia.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Alan Gray <alang@nvidia.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */

#include "gmxpre.h"

#include <math_constants.h>

#include <cassert>

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxassert.h"

#include "listed_forces_gpu_impl.h"

#if defined(_MSVC)
#    include <limits>
#endif


// \brief Staggered atomic force component accumulation to reduce clashes
//
// Reduce the number of atomic clashes by a theoretical max 3x by having consecutive threads
// accumulate different force components at the same time.
__device__ __forceinline__ void staggeredAtomicAddForce(float3* __restrict__ targetPtr, float3 f)
{
    int3 offset = make_int3(0, 1, 2);

    // Shift force components x, y, and z left by 2, 1, and 0, respectively
    // to end up with zxy, yzx, xyz on consecutive threads.
    f      = (threadIdx.x % 3 == 0) ? make_float3(f.y, f.z, f.x) : f;
    offset = (threadIdx.x % 3 == 0) ? make_int3(offset.y, offset.z, offset.x) : offset;
    f      = (threadIdx.x % 3 <= 1) ? make_float3(f.y, f.z, f.x) : f;
    offset = (threadIdx.x % 3 <= 1) ? make_int3(offset.y, offset.z, offset.x) : offset;

    atomicAdd(&targetPtr->x + offset.x, f.x);
    atomicAdd(&targetPtr->x + offset.y, f.y);
    atomicAdd(&targetPtr->x + offset.z, f.z);
}


/*-------------------------------- CUDA kernels-------------------------------- */
/*------------------------------------------------------------------------------*/

#define CUDA_DEG2RAD_F (CUDART_PI_F / 180.0F)

/*---------------- BONDED CUDA kernels--------------*/

/* Harmonic */
__device__ __forceinline__ static void
harmonic_gpu(const float kA, const float xA, const float x, float* V, float* F)
{
    constexpr float half = 0.5F;
    float           dx, dx2;

    dx  = x - xA;
    dx2 = dx * dx;

    *F = -kA * dx;
    *V = half * kA * dx2;
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void bonds_gpu(const int       i,
                                          float*          vtot_loc,
                                          const int       numBonds,
                                          const t_iatom   d_forceatoms[],
                                          const t_iparams d_forceparams[],
                                          const float4    gm_xq[],
                                          float3          gm_f[],
                                          float3          sm_fShiftLoc[],
                                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int3 bondData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = bondData.x;
        int        ai       = bondData.y;
        int        aj       = bondData.z;

        /* dx = xi - xj, corrected for periodic boundary conditions. */
        float3 dx;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

        float dr2 = norm2(dx);
        float dr  = sqrt(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.rA, dr, &vbond, &fbond);

        if (calcEner)
        {
            *vtot_loc += vbond;
        }

        if (dr2 != 0.0F)
        {
            fbond *= rsqrtf(dr2);

            float3 fij = fbond * dx;
            staggeredAtomicAddForce(&gm_f[ai], fij);
            staggeredAtomicAddForce(&gm_f[aj], -fij);
            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[ki], fij);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fij);
            }
        }
    }
}

template<bool returnShift>
__device__ __forceinline__ static float bond_angle_gpu(const float4   xi,
                                                       const float4   xj,
                                                       const float4   xk,
                                                       const PbcAiuc& pbcAiuc,
                                                       float3*        r_ij,
                                                       float3*        r_kj,
                                                       float*         costh,
                                                       int*           t1,
                                                       int*           t2)
/* Return value is the angle between the bonds i-j and j-k */
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, *r_kj);

    *costh   = cos_angle(*r_ij, *r_kj);
    float th = acosf(*costh);

    return th;
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void angles_gpu(const int       i,
                                           float*          vtot_loc,
                                           const int       numBonds,
                                           const t_iatom   d_forceatoms[],
                                           const t_iparams d_forceparams[],
                                           const float4    gm_xq[],
                                           float3          gm_f[],
                                           float3          sm_fShiftLoc[],
                                           const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int4 angleData = *(reinterpret_cast<const int4*>(d_forceatoms + 4 * i));
        int        type      = angleData.x;
        int        ai        = angleData.y;
        int        aj        = angleData.z;
        int        ak        = angleData.w;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(d_forceparams[type].harmonic.krA,
                     d_forceparams[type].harmonic.rA * CUDA_DEG2RAD_F,
                     theta,
                     &va,
                     &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0F)
        {
            float st    = dVdt * rsqrtf(1.0F - cos_theta2);
            float sth   = st * cos_theta;
            float nrij2 = norm2(r_ij);
            float nrkj2 = norm2(r_kj);

            float nrij_1 = rsqrtf(nrij2);
            float nrkj_1 = rsqrtf(nrkj2);

            float cik = st * nrij_1 * nrkj_1;
            float cii = sth * nrij_1 * nrij_1;
            float ckk = sth * nrkj_1 * nrkj_1;

            float3 f_i = cii * r_ij - cik * r_kj;
            float3 f_k = ckk * r_kj - cik * r_ij;
            float3 f_j = -f_i - f_k;

            staggeredAtomicAddForce(&gm_f[ai], f_i);
            staggeredAtomicAddForce(&gm_f[aj], f_j);
            staggeredAtomicAddForce(&gm_f[ak], f_k);

            if (calcVir)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[t1], f_i);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
                staggeredAtomicAddForce(&sm_fShiftLoc[t2], f_k);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void urey_bradley_gpu(const int       i,
                                                 float*          vtot_loc,
                                                 const int       numBonds,
                                                 const t_iatom   d_forceatoms[],
                                                 const t_iparams d_forceparams[],
                                                 const float4    gm_xq[],
                                                 float3          gm_f[],
                                                 float3          sm_fShiftLoc[],
                                                 const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int4 ubData = *(reinterpret_cast<const int4*>(d_forceatoms + 4 * i));
        int        type   = ubData.x;
        int        ai     = ubData.y;
        int        aj     = ubData.z;
        int        ak     = ubData.w;

        float th0A = d_forceparams[type].u_b.thetaA * CUDA_DEG2RAD_F;
        float kthA = d_forceparams[type].u_b.kthetaA;
        float r13A = d_forceparams[type].u_b.r13A;
        float kUBA = d_forceparams[type].u_b.kUBA;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(kthA, th0A, theta, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float3 r_ik;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

        float dr2 = norm2(r_ik);
        float dr  = dr2 * rsqrtf(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(kUBA, r13A, dr, &vbond, &fbond);

        float cos_theta2 = cos_theta * cos_theta;

        float3 f_i = make_float3(0.0F);
        float3 f_j = make_float3(0.0F);
        float3 f_k = make_float3(0.0F);

        if (cos_theta2 < 1.0F)
        {
            float st  = dVdt * rsqrtf(1.0F - cos_theta2);
            float sth = st * cos_theta;

            float nrkj2 = norm2(r_kj);
            float nrij2 = norm2(r_ij);

            float cik = st * rsqrtf(nrkj2 * nrij2);
            float cii = sth / nrij2;
            float ckk = sth / nrkj2;

            f_i = cii * r_ij - cik * r_kj;
            f_k = ckk * r_kj - cik * r_ij;
            f_j = -f_i - f_k;

            if (calcVir)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[t1], f_i);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
                staggeredAtomicAddForce(&sm_fShiftLoc[t2], f_k);
            }
        }

        /* Time for the bond calculations */
        if (dr2 != 0.0F)
        {
            if (calcEner)
            {
                *vtot_loc += vbond;
            }

            fbond *= rsqrtf(dr2);

            float3 fik = fbond * r_ik;
            f_i += fik;
            f_k -= fik;

            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[ki], fik);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fik);
            }
        }
        if ((cos_theta2 < 1.0F) || (dr2 != 0.0F))
        {
            staggeredAtomicAddForce(&gm_f[ai], f_i);
            staggeredAtomicAddForce(&gm_f[ak], f_k);
        }

        if (cos_theta2 < 1.0F)
        {
            staggeredAtomicAddForce(&gm_f[aj], f_j);
        }
    }
}

template<bool returnShift, typename T>
__device__ __forceinline__ static float dih_angle_gpu(const T        xi,
                                                      const T        xj,
                                                      const T        xk,
                                                      const T        xl,
                                                      const PbcAiuc& pbcAiuc,
                                                      float3*        r_ij,
                                                      float3*        r_kj,
                                                      float3*        r_kl,
                                                      float3*        m,
                                                      float3*        n,
                                                      int*           t1,
                                                      int*           t2,
                                                      int*           t3)
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, *r_kj);
    *t3 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xl, *r_kl);

    *m         = cprod(*r_ij, *r_kj);
    *n         = cprod(*r_kj, *r_kl);
    float phi  = gmx_angle(*m, *n);
    float ipr  = iprod(*r_ij, *n);
    float sign = (ipr < 0.0F) ? -1.0F : 1.0F;
    phi        = sign * phi;

    return phi;
}


__device__ __forceinline__ static void
dopdihs_gpu(const float cpA, const float phiA, const int mult, const float phi, float* v, float* f)
{
    float mdphi, sdphi;

    mdphi = mult * phi - phiA * CUDA_DEG2RAD_F;
    sdphi = sinf(mdphi);
    *v    = cpA * (1.0F + cosf(mdphi));
    *f    = -cpA * mult * sdphi;
}

template<bool calcVir>
__device__ __forceinline__ static void do_dih_fup_gpu(const int            i,
                                                      const int            j,
                                                      const int            k,
                                                      const int            l,
                                                      const float          ddphi,
                                                      const float3         r_ij,
                                                      const float3         r_kj,
                                                      const float3         r_kl,
                                                      const float3         m,
                                                      const float3         n,
                                                      float3               gm_f[],
                                                      float3               sm_fShiftLoc[],
                                                      const PbcAiuc&       pbcAiuc,
                                                      const float4         gm_xq[],
                                                      const int            t1,
                                                      const int            t2,
                                                      const int gmx_unused t3)
{
    float iprm  = norm2(m);
    float iprn  = norm2(n);
    float nrkj2 = norm2(r_kj);
    float toler = nrkj2 * GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        float  nrkj_1 = rsqrtf(nrkj2); // replacing std::invsqrt call
        float  nrkj_2 = nrkj_1 * nrkj_1;
        float  nrkj   = nrkj2 * nrkj_1;
        float  a      = -ddphi * nrkj / iprm;
        float3 f_i    = a * m;
        float  b      = ddphi * nrkj / iprn;
        float3 f_l    = b * n;
        float  p      = iprod(r_ij, r_kj);
        p *= nrkj_2;
        float q = iprod(r_kl, r_kj);
        q *= nrkj_2;
        float3 uvec = p * f_i;
        float3 vvec = q * f_l;
        float3 svec = uvec - vvec;
        float3 f_j  = f_i - svec;
        float3 f_k  = f_l + svec;

        staggeredAtomicAddForce(&gm_f[i], f_i);
        staggeredAtomicAddForce(&gm_f[j], -f_j);
        staggeredAtomicAddForce(&gm_f[k], -f_k);
        staggeredAtomicAddForce(&gm_f[l], f_l);

        if (calcVir)
        {
            float3 dx_jl;
            int    t3 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[l], gm_xq[j], dx_jl);

            staggeredAtomicAddForce(&sm_fShiftLoc[t1], f_i);
            staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -f_j);
            staggeredAtomicAddForce(&sm_fShiftLoc[t2], -f_k);
            staggeredAtomicAddForce(&sm_fShiftLoc[t3], f_l);
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void pdihs_gpu(const int       i,
                                          float*          vtot_loc,
                                          const int       numBonds,
                                          const t_iatom   d_forceatoms[],
                                          const t_iparams d_forceparams[],
                                          const float4    gm_xq[],
                                          float3          gm_f[],
                                          float3          sm_fShiftLoc[],
                                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        float vpd;
        float ddphi;
        dopdihs_gpu(d_forceparams[type].pdihs.cpA,
                    d_forceparams[type].pdihs.phiA,
                    d_forceparams[type].pdihs.mult,
                    phi,
                    &vpd,
                    &ddphi);

        if (calcEner)
        {
            *vtot_loc += vpd;
        }

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void rbdihs_gpu(const int       i,
                                           float*          vtot_loc,
                                           const int       numBonds,
                                           const t_iatom   d_forceatoms[],
                                           const t_iparams d_forceparams[],
                                           const float4    gm_xq[],
                                           float3          gm_f[],
                                           float3          sm_fShiftLoc[],
                                           const PbcAiuc   pbcAiuc)
{
    constexpr float c0 = 0.0F, c1 = 1.0F, c2 = 2.0F, c3 = 3.0F, c4 = 4.0F, c5 = 5.0F;

    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* Change to polymer convention */
        if (phi < c0)
        {
            phi += CUDART_PI_F;
        }
        else
        {
            phi -= CUDART_PI_F;
        }
        float cos_phi = cosf(phi);
        /* Beware of accuracy loss, cannot use 1-sqrt(cos^2) ! */
        float sin_phi = sinf(phi);

        float parm[NR_RBDIHS];
        for (int j = 0; j < NR_RBDIHS; j++)
        {
            parm[j] = d_forceparams[type].rbdihs.rbcA[j];
        }
        /* Calculate cosine powers */
        /* Calculate the energy */
        /* Calculate the derivative */
        float v      = parm[0];
        float ddphi  = c0;
        float cosfac = c1;

        float rbp = parm[1];
        ddphi += rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[2];
        ddphi += c2 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[3];
        ddphi += c3 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[4];
        ddphi += c4 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[5];
        ddphi += c5 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }

        ddphi = -ddphi * sin_phi;

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
        if (calcEner)
        {
            *vtot_loc += v;
        }
    }
}

__device__ __forceinline__ static void make_dp_periodic_gpu(float* dp)
{
    /* dp cannot be outside (-pi,pi) */
    if (*dp >= CUDART_PI_F)
    {
        *dp -= 2.0F * CUDART_PI_F;
    }
    else if (*dp < -CUDART_PI_F)
    {
        *dp += 2.0F * CUDART_PI_F;
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void idihs_gpu(const int       i,
                                          float*          vtot_loc,
                                          const int       numBonds,
                                          const t_iatom   d_forceatoms[],
                                          const t_iparams d_forceparams[],
                                          const float4    gm_xq[],
                                          float3          gm_f[],
                                          float3          sm_fShiftLoc[],
                                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* phi can jump if phi0 is close to Pi/-Pi, which will cause huge
         * force changes if we just apply a normal harmonic.
         * Instead, we first calculate phi-phi0 and take it modulo (-Pi,Pi).
         * This means we will never have the periodicity problem, unless
         * the dihedral is Pi away from phiO, which is very unlikely due to
         * the potential.
         */
        float kA = d_forceparams[type].harmonic.krA;
        float pA = d_forceparams[type].harmonic.rA;

        float phi0 = pA * CUDA_DEG2RAD_F;

        float dp = phi - phi0;

        make_dp_periodic_gpu(&dp);

        float ddphi = -kA * dp;

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);

        if (calcEner)
        {
            *vtot_loc += -0.5F * ddphi * dp;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void pairs_gpu(const int       i,
                                          const int       numBonds,
                                          const t_iatom   d_forceatoms[],
                                          const t_iparams iparams[],
                                          const float4    gm_xq[],
                                          float3          gm_f[],
                                          float3          sm_fShiftLoc[],
                                          const PbcAiuc   pbcAiuc,
                                          const float     scale_factor,
                                          float*          vtotVdw_loc,
                                          float*          vtotElec_loc,
                                          const int       pType)
{
    if (i < numBonds)
    {
        // TODO this should be made into a separate type, the GPU and CPU sizes should be compared
        const int3 pairData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = pairData.x;
        int        ai       = pairData.y;
        int        aj       = pairData.z;
        float      qq = 0.0F;
        float      c6 = 0.0F;
        float      c12 = 0.0F;

        // F_LJ14
        if (pType == 0) {
            qq  = gm_xq[ai].w * gm_xq[aj].w;
            c6  = iparams[type].lj14.c6A;
            c12 = iparams[type].lj14.c12A;
        }
        // F_LJC14_Q
        else if (pType == 1)
        {
            qq  = iparams[type].ljc14.qi * iparams[type].ljc14.qj * iparams[type].ljc14.fqq;
            c6  = iparams[type].ljc14.c6;
            c12 = iparams[type].ljc14.c12;
        }
        //F_LJC_PAIRS_NB
        else if (pType == 2)
        {
            qq  = iparams[type].ljcnb.qi * iparams[type].ljcnb.qj;
            c6  = iparams[type].ljcnb.c6;
            c12 = iparams[type].ljcnb.c12;
        }
        /* Do we need to apply full periodic boundary conditions? */
        float3 dr;
        int    fshift_index = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

        float r2    = norm2(dr);
        float rinv  = rsqrtf(r2);
        float rinv2 = rinv * rinv;
        float rinv6 = rinv2 * rinv2 * rinv2;

        /* Calculate the Coulomb force * r */
        float velec = scale_factor * qq * rinv;

        /* Calculate the LJ force * r and add it to the Coulomb part */
        float fr = (12.0F * c12 * rinv6 - 6.0F * c6) * rinv6 + velec;

        float  finvr = fr * rinv2;
        float3 f     = finvr * dr;

        /* Add the forces */
        staggeredAtomicAddForce(&gm_f[ai], f);
        staggeredAtomicAddForce(&gm_f[aj], -f);
        if (calcVir && fshift_index != gmx::c_centralShiftIndex)
        {
            staggeredAtomicAddForce(&sm_fShiftLoc[fshift_index], f);
            staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -f);
        }

        if (calcEner)
        {
            *vtotVdw_loc += (c12 * rinv6 - c6) * rinv6;
            *vtotElec_loc += velec;
        }
    }
}

// FEP GPU kernels
__device__ __forceinline__ static float
harmonic_fep_gpu(const float kA, const float kB, const float xA, const float xB, const float x, const float lambda, float* V, float* F)
{
    constexpr float half = 0.5F;
    float           L1, kk, x0, dx, dx2;
    float           v, f;
    float           dvdlambda = 0.0F;

    L1 = 1.0F - lambda;
    kk = L1 * kA + lambda * kB;
    x0 = L1 * xA + lambda * xB;

    dx  = x - x0;
    dx2 = dx * dx;

    f         = -kk * dx;
    v         = half * kk * dx2;

    if ((kA != kB) || (xA != xB))
        dvdlambda = half * (kB - kA) * dx2 + (xA - xB) * kk * dx;

    *F = f;
    *V = v;

    return dvdlambda;
}

template<bool calcVir, bool calcEner>
__device__ void bonds_fep_gpu(const int                 i,
                              float*                    vtot_loc,
                              const int                 numBonds,
                              const t_iatom             d_forceatoms[],
                              const t_iparams           d_forceparams[],
                              gmx::BondedFepParameters* d_fepparams,
                              const float4              gm_xq[],
                              float3                      gm_f[],
                              float3                      sm_fShiftLoc[],
                              const PbcAiuc             pbcAiuc,
                              float*                    dvdltot_loc)
{
    if (i < numBonds)
    {
        const int3 bondData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = bondData.x;
        int        ai       = bondData.y;
        int        aj       = bondData.z;

        /* dx = xi - xj, corrected for periodic boundary conditions. */
        float3 dx;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

        float dr2 = norm2(dx);
        float dr  = sqrt(dr2);

        float vbond;
        float fbond;
        float dvdlambda;

        dvdlambda = harmonic_fep_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.krB,
            d_forceparams[type].harmonic.rA, d_forceparams[type].harmonic.rB, dr,
            d_fepparams->lambdaBonded, &vbond, &fbond);

        if (calcEner)
        {
            *vtot_loc += vbond;
            *dvdltot_loc += dvdlambda;
        }

        if (dr2 != 0.0F)
        {
            fbond *= rsqrtf(dr2);

            float3 fij = fbond * dx;
            staggeredAtomicAddForce(&gm_f[ai], fij);
            staggeredAtomicAddForce(&gm_f[aj], -fij);
            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[ki], fij);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fij);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void angles_fep_gpu(const int       i,
                                           float*          vtot_loc,
                                           const int       numBonds,
                                           const t_iatom   d_forceatoms[],
                                           const t_iparams d_forceparams[],
                                           gmx::BondedFepParameters* d_fepparams,
                                           const float4    gm_xq[],
                                           float3          gm_f[],
                                           float3          sm_fShiftLoc[],
                                           const PbcAiuc   pbcAiuc,
                                           float*          dvdltot_loc)
{
    if (i < numBonds)
    {
        const int4 angleData = *(reinterpret_cast<const int4*>(d_forceatoms + 4 * i));
        int        type      = angleData.x;
        int        ai        = angleData.y;
        int        aj        = angleData.z;
        int        ak        = angleData.w;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        float dvdlambda;

        dvdlambda = harmonic_fep_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.krB,
            d_forceparams[type].harmonic.rA * CUDA_DEG2RAD_F,
            d_forceparams[type].harmonic.rB * CUDA_DEG2RAD_F, theta,
            d_fepparams->lambdaBonded, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
            *dvdltot_loc += dvdlambda;
        }

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0F)
        {
            float st    = dVdt * rsqrtf(1.0F - cos_theta2);
            float sth   = st * cos_theta;
            float nrij2 = norm2(r_ij);
            float nrkj2 = norm2(r_kj);

            float nrij_1 = rsqrtf(nrij2);
            float nrkj_1 = rsqrtf(nrkj2);

            float cik = st * nrij_1 * nrkj_1;
            float cii = sth * nrij_1 * nrij_1;
            float ckk = sth * nrkj_1 * nrkj_1;

            float3 f_i = cii * r_ij - cik * r_kj;
            float3 f_k = ckk * r_kj - cik * r_ij;
            float3 f_j = -f_i - f_k;

            staggeredAtomicAddForce(&gm_f[ai], f_i);
            staggeredAtomicAddForce(&gm_f[aj], f_j);
            staggeredAtomicAddForce(&gm_f[ak], f_k);

            if (calcVir)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[t1], f_i);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
                staggeredAtomicAddForce(&sm_fShiftLoc[t2], f_k);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void urey_bradley_fep_gpu(const int       i,
                                                 float*          vtot_loc,
                                                 const int       numBonds,
                                                 const t_iatom   d_forceatoms[],
                                                 const t_iparams d_forceparams[],
                                                 gmx::BondedFepParameters* d_fepparams,
                                                 const float4    gm_xq[],
                                                 float3          gm_f[],
                                                 float3          sm_fShiftLoc[],
                                                 const PbcAiuc   pbcAiuc,
                                                 float*          dvdltot_loc)
{
    if (i < numBonds)
    {
        const int4 ubData = *(reinterpret_cast<const int4*>(d_forceatoms + 4 * i));
        int        type   = ubData.x;
        int        ai     = ubData.y;
        int        aj     = ubData.z;
        int        ak     = ubData.w;

        float th0A = d_forceparams[type].u_b.thetaA * CUDA_DEG2RAD_F;
        float kthA = d_forceparams[type].u_b.kthetaA;
        float r13A = d_forceparams[type].u_b.r13A;
        float kUBA = d_forceparams[type].u_b.kUBA;

        float th0B = d_forceparams[type].u_b.thetaB * CUDA_DEG2RAD_F;
        float kthB = d_forceparams[type].u_b.kthetaB;
        float r13B = d_forceparams[type].u_b.r13B;
        float kUBB = d_forceparams[type].u_b.kUBB;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        float dvdlambda;

        dvdlambda = harmonic_fep_gpu(kthA, kthB, th0A, th0B, theta, d_fepparams->lambdaBonded, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
            *dvdltot_loc += dvdlambda;
        }

        float3 r_ik;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

        float dr2 = norm2(r_ik);
        float dr  = dr2 * rsqrtf(dr2);

        float vbond;
        float fbond;

        dvdlambda = harmonic_fep_gpu(kUBA, kUBB, r13A, r13B, dr, d_fepparams->lambdaBonded, &vbond, &fbond);

        float cos_theta2 = cos_theta * cos_theta;

        float3 f_i = make_float3(0.0F);
        float3 f_j = make_float3(0.0F);
        float3 f_k = make_float3(0.0F);

        if (cos_theta2 < 1.0F)
        {
            float st  = dVdt * rsqrtf(1.0F - cos_theta2);
            float sth = st * cos_theta;

            float nrkj2 = norm2(r_kj);
            float nrij2 = norm2(r_ij);

            float cik = st * rsqrtf(nrkj2 * nrij2);
            float cii = sth / nrij2;
            float ckk = sth / nrkj2;

            f_i = cii * r_ij - cik * r_kj;
            f_k = ckk * r_kj - cik * r_ij;
            f_j = -f_i - f_k;

            if (calcVir)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[t1], f_i);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
                staggeredAtomicAddForce(&sm_fShiftLoc[t2], f_k);
            }
        }

        /* the bond calculations */
        if (dr2 != 0.0F)
        {
            if (calcEner)
            {
                *vtot_loc += vbond;
                *dvdltot_loc += dvdlambda;
            }

            fbond *= rsqrtf(dr2);

            float3 fik = fbond * r_ik;
            f_i += fik;
            f_k -= fik;

            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[ki], fik);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fik);
            }
        }
        if ((cos_theta2 < 1.0F) || (dr2 != 0.0F))
        {
            staggeredAtomicAddForce(&gm_f[ai], f_i);
            staggeredAtomicAddForce(&gm_f[ak], f_k);
        }

        if (cos_theta2 < 1.0F)
        {
            staggeredAtomicAddForce(&gm_f[aj], f_j);
        }
    }
}

__device__ __forceinline__ static float
dopdihs_fep_gpu(const float cpA, const float cpB, const float phiA, const float phiB, const int mult, const float phi, const float lambda, float* v, float* f)
{
    float mdphi, v1, sdphi, ddphi;
    float dvdlambda = 0.0F;
    float L1 = 1.0F - lambda;
    float phi0 = (L1 * phiA + lambda * phiB) * CUDA_DEG2RAD_F;
    float dph0 = (phiB - phiA) * CUDA_DEG2RAD_F;
    float cp = L1 * cpA + lambda * cpB;

    mdphi = mult * phi - phi0;
    sdphi = sinf(mdphi);
    ddphi = -cp * mult * sdphi;
    v1 = 1.0F + cosf(mdphi);

    if ((cpA != cpB) || (phiA != phiB))
        dvdlambda = (cpB - cpA) * v1 + cp * dph0 * sdphi;

    *v    = cp * v1;
    *f    = ddphi;

    return dvdlambda;
}

__device__ __forceinline__ static float
dopdihs_min_fep_gpu(const float cpA, const float cpB, const float phiA, const float phiB, const int mult, const float phi, const float lambda, float* v, float* f)
{
    float dvdlambda, mdphi, v1, sdphi, ddphi;
    float L1 = 1.0F - lambda;
    float phi0 = (L1 * phiA + lambda * phiB) * CUDA_DEG2RAD_F;
    float dph0 = (phiB - phiA) * CUDA_DEG2RAD_F;
    float cp = L1 * cpA + lambda * cpB;

    mdphi = mult * (phi - phi0);
    sdphi = sinf(mdphi);
    ddphi = cp * mult * sdphi;
    v1 = 1.0F - cosf(mdphi);

    dvdlambda = (cpB - cpA) * v1 + cp * dph0 * sdphi;

    *v    = cp * v1;
    *f    = ddphi;

    return dvdlambda;
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void pdihs_fep_gpu(const int       i,
                                          float*          vtot_loc,
                                          const int       numBonds,
                                          const t_iatom   d_forceatoms[],
                                          const t_iparams d_forceparams[],
                                          gmx::BondedFepParameters* d_fepparams,
                                          const float4    gm_xq[],
                                          float3          gm_f[],
                                          float3          sm_fShiftLoc[],
                                          const PbcAiuc   pbcAiuc,
                                          float*          dvdltot_loc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        float vpd;
        float ddphi;
        float dvdlambda;

        dvdlambda = dopdihs_fep_gpu(d_forceparams[type].pdihs.cpA, d_forceparams[type].pdihs.cpB,
                    d_forceparams[type].pdihs.phiA, d_forceparams[type].pdihs.phiB,
                    d_forceparams[type].pdihs.mult,
                    phi, d_fepparams->lambdaBonded,
                    &vpd,
                    &ddphi);

        if (calcEner)
        {
            *vtot_loc += vpd;
            *dvdltot_loc += dvdlambda;
        }

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void rbdihs_fep_gpu(const int       i,
                                           float*          vtot_loc,
                                           const int       numBonds,
                                           const t_iatom   d_forceatoms[],
                                           const t_iparams d_forceparams[],
                                           gmx::BondedFepParameters* d_fepparams,
                                           const float4    gm_xq[],
                                           float3          gm_f[],
                                           float3          sm_fShiftLoc[],
                                           const PbcAiuc   pbcAiuc,
                                           float*          dvdltot_loc)
{
    constexpr float c0 = 0.0F, c1 = 1.0F, c2 = 2.0F, c3 = 3.0F, c4 = 4.0F, c5 = 5.0F;
    float dvdlambda = 0.0F;

    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        float lambda = d_fepparams->lambdaBonded;
        /* Change to polymer convention */
        if (phi < c0)
        {
            phi += CUDART_PI_F;
        }
        else
        {
            phi -= CUDART_PI_F;
        }
        float cos_phi = cosf(phi);
        /* Beware of accuracy loss, cannot use 1-sqrt(cos^2) ! */
        float sin_phi = sinf(phi);

        float parm[NR_RBDIHS];
        float parmA[NR_RBDIHS];
        float parmB[NR_RBDIHS];
        for (int j = 0; j < NR_RBDIHS; j++)
        {
            parmA[j] = d_forceparams[type].rbdihs.rbcA[j];
            parmB[j] = d_forceparams[type].rbdihs.rbcB[j];
            parm[j] = (1-lambda) * parmA[j] + lambda * parmB[j];
        }
        /* Calculate cosine powers */
        /* Calculate the energy */
        /* Calculate the derivative */
        float v      = parm[0];
        if (parmA[0] != parmB[0])
            dvdlambda += (parmB[0] - parmA[0]);
        float ddphi  = c0;
        float cosfac = c1;

        float rbp = parm[1];
        ddphi += rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
            if (parmA[1] != parmB[1])
                dvdlambda += cosfac * (parmB[1] - parmA[1]);
        }
        rbp = parm[2];
        ddphi += c2 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
            if (parmA[2] != parmB[2])
                dvdlambda += cosfac * (parmB[2] - parmA[2]);
        }
        rbp = parm[3];
        ddphi += c3 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
            if (parmA[3] != parmB[3])
                dvdlambda += cosfac * (parmB[3] - parmA[3]);
        }
        rbp = parm[4];
        ddphi += c4 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
            if (parmA[4] != parmB[4])
                dvdlambda += cosfac * (parmB[4] - parmA[4]);
        }
        rbp = parm[5];
        ddphi += c5 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
            if (parmA[5] != parmB[5])
                dvdlambda += cosfac * (parmB[5] - parmA[5]);
        }

        ddphi = -ddphi * sin_phi;

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
        if (calcEner)
        {
            *vtot_loc += v;
            *dvdltot_loc += dvdlambda;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void idihs_fep_gpu(const int       i,
                                          float*          vtot_loc,
                                          const int       numBonds,
                                          const t_iatom   d_forceatoms[],
                                          const t_iparams d_forceparams[],
                                          gmx::BondedFepParameters* d_fepparams,
                                          const float4    gm_xq[],
                                          float3          gm_f[],
                                          float3          sm_fShiftLoc[],
                                          const PbcAiuc   pbcAiuc,
                                          float*          dvdltot_loc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float dvdlambda = 0.0F;
        float lambda = d_fepparams->lambdaBonded;
        float L1        = 1.0F - lambda;

        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* phi can jump if phi0 is close to Pi/-Pi, which will cause huge
         * force changes if we just apply a normal harmonic.
         * Instead, we first calculate phi-phi0 and take it modulo (-Pi,Pi).
         * This means we will never have the periodicity problem, unless
         * the dihedral is Pi away from phiO, which is very unlikely due to
         * the potential.
         */
        float kA = d_forceparams[type].harmonic.krA;
        float pA = d_forceparams[type].harmonic.rA;
        float kB = d_forceparams[type].harmonic.krB;
        float pB = d_forceparams[type].harmonic.rB;

        float kk = L1 * kA + lambda * kB;
        float phi0  = (L1 * pA + lambda * pB) * CUDA_DEG2RAD_F;
        float dphi0 = (pB - pA) * CUDA_DEG2RAD_F;

        float dp = phi - phi0;
        make_dp_periodic_gpu(&dp);
        float dp2 = dp * dp;

        float ddphi = -kk * dp;

        if ((kB != kA) || (pA != pB))
            dvdlambda += 0.5F * (kB - kA) * dp2 - kk * dphi0 * dp;

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);

        if (calcEner)
        {
            *vtot_loc += 0.5F * kk * dp2;
            *dvdltot_loc += dvdlambda;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void pairs_fep_gpu(const int       i,
                                          const int       numBonds,
                                          const t_iatom   d_forceatoms[],
                                          const t_iparams iparams[],
                                          gmx::BondedFepParameters* d_fepparams,
                                          const float4    gm_xq[],
                                          const float4     gm_q4[],
                                          float3          gm_f[],
                                          float3          sm_fShiftLoc[],
                                          const PbcAiuc   pbcAiuc,
                                          const float     scale_factor,
                                          float*          vtotVdw_loc,
                                          float*          vtotElec_loc,
                                          float*          dvdltotVdw_loc,
                                          float*          dvdltotElec_loc)
{
    bool       bFEPpair;
    const float alphaCoulomb = d_fepparams->alphaCoul;
    const float alphaVdw  = d_fepparams->alphaVdw;
    float alphaCoulombEff   = alphaCoulomb;
    float alphaVdwEff  = alphaVdw;
    const bool useSoftCore = (alphaVdw != 0.0F);
    const float sigma6_def = d_fepparams->sc_sigma6;
    const float sigma6_min = d_fepparams->sc_sigma6_min;
    const float lambdaCoul   = d_fepparams->lambdaCoul;
    const float _lambdaCoul  = 1.0F - lambdaCoul;
    const float lambdaVdw   = d_fepparams->lambdaVdw;
    const float _lambdaVdw  = 1.0F - lambdaVdw;
    const int     lambdaPower   = d_fepparams->lambdaPower;

    float lambdaFactorCoul[2]       = { _lambdaCoul, lambdaCoul };
    float lambdaFactorVdw[2]       = { _lambdaVdw, lambdaVdw };
    float softcoreLambdaFactorCoul[2] = { lambdaCoul, _lambdaCoul };
    float softcoreLambdaFactorVdw[2]  = { lambdaVdw, _lambdaVdw };

    float dLambdaFactor[2];
    float softcoreDlFactorCoul[2];
    float softcoreDlFactorVdw[2];

    if (calcEner) {

        /*derivative of the lambda factor for state A and B */
        dLambdaFactor[0] = -1.0F;
        dLambdaFactor[1] = 1.0F;

        constexpr float softcoreRPower = 6.0F;

        for (int k = 0; k < 2; k++)
        {
            softcoreLambdaFactorCoul[k] =
                    (lambdaPower == 2 ? (1.0F - lambdaFactorCoul[k]) * (1.0F - lambdaFactorCoul[k])
                                    : (1.0F - lambdaFactorCoul[k]));
            softcoreDlFactorCoul[k] = dLambdaFactor[k] * lambdaPower / softcoreRPower
                                    * (lambdaPower == 2 ? (1.0F - lambdaFactorCoul[k]) : 1.0F);
            softcoreLambdaFactorVdw[k] =
                    (lambdaPower == 2 ? (1.0F - lambdaFactorVdw[k]) * (1.0F - lambdaFactorVdw[k])
                                    : (1.0F - lambdaFactorVdw[k]));
            softcoreDlFactorVdw[k] = dLambdaFactor[k] * lambdaPower / softcoreRPower
                                    * (lambdaPower == 2 ? (1.0F - lambdaFactorVdw[k]) : 1.0F);
        }
    }

    float scalarForcePerDistanceCoul[2], scalarForcePerDistanceVdw[2];
    float Vcoul[2], Vvdw[2];
    float rInvC, r2C, rPInvC, rPInvV;
    float qq[2], c6AB[2], c12AB[2];
    float4 q4_i, q4_j;

    if (i < numBonds)
    {
        // TODO this should be made into a separate type, the GPU and CPU sizes should be compared
        const int3 pairData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = pairData.x;
        int        ai       = pairData.y;
        int        aj       = pairData.z;

        q4_i = gm_q4[ai];
        q4_j = gm_q4[aj];

        qq[0] = q4_i.x * q4_j.x;
        qq[1] = q4_i.y * q4_j.y;
        c6AB[0] = iparams[type].lj14.c6A;
        c6AB[1] = iparams[type].lj14.c6B;
        c12AB[0] = iparams[type].lj14.c12A;
        c12AB[1] = iparams[type].lj14.c12B;

        float sigma6[2];
        float velec = 0.0F;
        float vlj   = 0.0F;
        float finvr = 0.0F;
        float dvdl_elec = 0.0F;
        float dvdl_lj   = 0.0F;

        if (qq[0] == qq[1] && c6AB[0] == c6AB[1] && c12AB[0] == c12AB[1]) bFEPpair = 0;
        else bFEPpair = 1;

        /* Do we need to apply full periodic boundary conditions? */
        float3 dr;
        int    fshift_index = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

        float r2    = norm2(dr);
        float rpm2  = r2 * r2;
        float rp    = rpm2 * r2;
        float rInv  = rsqrtf(r2);
        float rInv2 = rInv * rInv;
        float rInv6 = rInv2 * rInv2 * rInv2;

        if (bFEPpair)
        {
            if (useSoftCore)
            {
                if ((c12AB[0] > 0.0F) && (c12AB[1] > 0.0F))
                {
                    alphaVdwEff  = 0.0F;
                    alphaCoulombEff = 0.0F;
                }
                else
                {
                    alphaVdwEff  = alphaVdw;
                    alphaCoulombEff = alphaCoulomb;
                }
            }

            for (int k = 0; k < 2; k++)
            {
                scalarForcePerDistanceCoul[k] = 0.0F;
                scalarForcePerDistanceVdw[k] = 0.0F;
                if (calcEner)
                {
                    Vcoul[k] = 0.0F;
                    Vvdw[k]  = 0.0F;
                }

                if ((qq[k] != 0.0F) || (c6AB[k] != 0.0F) || (c12AB[k] != 0.0F))
                {
                    if ((c12AB[0] == 0.0F || c12AB[1] == 0.0F) && (useSoftCore))
                    {
                        if (c6AB[k] == 0.0F)
                            sigma6[k] = 0.0F;
                        else
                            sigma6[k] = c12AB[k] / c6AB[k];

                        if (sigma6[k] == 0.0F)
                            sigma6[k] = sigma6_def;
                        if (sigma6[k] < sigma6_min)
                            sigma6[k] = sigma6_min;

                        rPInvC = 1.0F / (alphaCoulombEff * softcoreLambdaFactorCoul[k] * sigma6[k] + rp);
                        r2C   = rcbrt(rPInvC);
                        rInvC = rsqrt(r2C);

                        if ((alphaCoulombEff != alphaVdwEff) || (softcoreLambdaFactorVdw[k] != softcoreLambdaFactorCoul[k]))
                        {
                            rPInvV = 1.0F / (alphaVdwEff * softcoreLambdaFactorVdw[k] * sigma6[k] + rp);
                        }
                        else
                        {
                            /* We can avoid one expensive pow and one / operation */
                            rPInvV = rPInvC;
                        }
                    }
                    else
                    {
                        rPInvC = rInv6;
                        rInvC  = rInv;
                        rPInvV = rInv6;
                    }

                    if (c6AB[k] != 0.0F || c12AB[k] != 0.0F)
                    {
                        float Vvdw6  = c6AB[k] * rPInvV;
                        float Vvdw12 = c12AB[k] * rPInvV * rPInvV;
                        scalarForcePerDistanceVdw[k]    = 12.0F * Vvdw12 - 6.0F * Vvdw6;
                        if (calcEner)
                        {
                            Vvdw[k] = Vvdw12 - Vvdw6;
                        }
                    }

                    if (qq[k] != 0.0F)
                    {
                        scalarForcePerDistanceCoul[k] = scale_factor * qq[k] * rInvC;
                        Vcoul[k]  = scalarForcePerDistanceCoul[k];
                    }
                    scalarForcePerDistanceCoul[k] *= rPInvC;
                    scalarForcePerDistanceVdw[k] *= rPInvV;
                }
            }
            for (int k = 0; k < 2; k++)
            {
                if (calcEner)
                {
                    velec += lambdaFactorCoul[k] * Vcoul[k];
                    vlj += lambdaFactorVdw[k] * Vvdw[k];
                    dvdl_elec += Vcoul[k] * dLambdaFactor[k];
                    dvdl_lj += Vvdw[k] * dLambdaFactor[k];
                    if (useSoftCore)
                    {
                        dvdl_elec += lambdaFactorCoul[k] * alphaCoulombEff * softcoreDlFactorCoul[k] * scalarForcePerDistanceCoul[k] * sigma6[k];
                        dvdl_lj += lambdaFactorVdw[k] * alphaVdwEff * softcoreDlFactorVdw[k] * scalarForcePerDistanceVdw[k] * sigma6[k];
                    }
                }
                finvr += lambdaFactorCoul[k] * scalarForcePerDistanceCoul[k] * rpm2;
                finvr += lambdaFactorVdw[k] * scalarForcePerDistanceVdw[k] * rpm2;
            }
        }
        else
        {
            /* Calculate the Coulomb force * r */
            velec = scale_factor * qq[0] * rInv;
            vlj   = (c12AB[0] * rInv6 - c6AB[0]) * rInv6;

            /* Calculate the LJ force * r and add it to the Coulomb part */
            float fr = (12.0F * c12AB[0] * rInv6 - 6.0F * c6AB[0]) * rInv6 + velec;
            finvr    = fr * rInv2;
        }

        float3 f     = finvr * dr;

        /* Add the forces */
        staggeredAtomicAddForce(&gm_f[ai], f);
        staggeredAtomicAddForce(&gm_f[aj], -f);
        if (calcVir && fshift_index != gmx::c_centralShiftIndex)
        {
            staggeredAtomicAddForce(&sm_fShiftLoc[fshift_index], f);
            staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -f);
        }

        if (calcEner)
        {
            *vtotVdw_loc += vlj;
            *vtotElec_loc += velec;
            *dvdltotVdw_loc += dvdl_lj;
            *dvdltotElec_loc += dvdl_elec;
        }
    }
}

// Restrict gpu kernels

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void restraint_bonds_gpu(const int                 i,
                                                    float*                    vtot_loc,
                                                    const int                 numBonds,
                                                    const t_iatom             d_forceatoms[],
                                                    const t_iparams           d_forceparams[],
                                                    const float               lambda_restr,
                                                    const float4              gm_xq[],
                                                    float3                    gm_f[],
                                                    float3                    sm_fShiftLoc[],
                                                    const PbcAiuc             pbcAiuc,
                                                    float*                    dvdltot_loc)
{
    if (i < numBonds)
    {
        const int3 bondData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = bondData.x;
        int        ai       = bondData.y;
        int        aj       = bondData.z;

        //float lambda_restr = d_fepparams->lambdaRestrict;
        float L1 = 1.0F - lambda_restr;

        float3 dx;
        int   ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

        float dr2 = norm2(dx);
        float dr  = sqrt(dr2);

        float fbond, vbond, dvdlambda;
        float drh, drh2;

        float low  = L1 * d_forceparams[type].restraint.lowA + lambda_restr * d_forceparams[type].restraint.lowB;
        float dlow = -d_forceparams[type].restraint.lowA + d_forceparams[type].restraint.lowB;
        float up1  = L1 * d_forceparams[type].restraint.up1A + lambda_restr * d_forceparams[type].restraint.up1B;
        float dup1 = -d_forceparams[type].restraint.up1A + d_forceparams[type].restraint.up1B;
        float up2  = L1 * d_forceparams[type].restraint.up2A + lambda_restr * d_forceparams[type].restraint.up2B;
        float dup2 = -d_forceparams[type].restraint.up2A + d_forceparams[type].restraint.up2B;
        float k    = L1 * d_forceparams[type].restraint.kA + lambda_restr * d_forceparams[type].restraint.kB;
        float dk   = -d_forceparams[type].restraint.kA + d_forceparams[type].restraint.kB;

        if (dr < low)
        {
            drh   = dr - low;
            drh2  = drh * drh;
            vbond = 0.5F * k * drh2;
            fbond = -k * drh;
            dvdlambda = 0.5F * dk * drh2 - k * dlow * drh;
        } 
        else if (dr <= up1)
        {
            vbond = 0.0F;
            fbond = 0.0F;
        }
        else if (dr <= up2)
        {
            drh   = dr - up1;
            drh2  = drh * drh;
            vbond = 0.5F * k * drh2;
            fbond = -k * drh;
            dvdlambda = 0.5F * dk * drh2 - k * dup1 * drh;
        }
        else
        {
            drh   = dr - up2;
            vbond = k * (up2 - up1) * (0.5F * (up2 - up1) + drh);
            fbond = -k * (up2 - up1);
            dvdlambda = dk * (up2 - up1) * (0.5F * (up2 - up1) + drh)
                          + k * (dup2 - dup1) * (up2 - up1 + drh) - k * (up2 - up1) * dup2;
        }

        if (calcEner)
        {
            *vtot_loc += vbond;
            *dvdltot_loc += dvdlambda;
        }

        if (dr2 != 0.0F)
        {
            fbond *= rsqrtf(dr2);

            float3 fij = fbond * dx;
            staggeredAtomicAddForce(&gm_f[ai], fij);
            staggeredAtomicAddForce(&gm_f[aj], -fij);
            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[ki], fij);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fij);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void angleres_gpu(const int       i,
                                            float*          vtot_loc,
                                            const int       numBonds,
                                            const t_iatom   d_forceatoms[],
                                            const t_iparams d_forceparams[],
                                            const float     lambda_restr,
                                            const float4    gm_xq[],
                                            float3          gm_f[],
                                            float3          sm_fShiftLoc[],
                                            const PbcAiuc   pbcAiuc,
                                            float           *dvdltot_loc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kl;

        int t1 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[aj], gm_xq[ai], r_ij);
        int t2 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[al], gm_xq[ak], r_kl);

        float cos_phi = cos_angle(r_ij, r_kl);
        float phi     = acosf(cos_phi);
        float vid, dVdphi, dvdlambda = 0.0F;
        float st, sth, nrij2, nrkl2, c, cij, ckl;

        dvdlambda = dopdihs_min_fep_gpu(d_forceparams[type].pdihs.cpA,
                                  d_forceparams[type].pdihs.cpB,
                                  d_forceparams[type].pdihs.phiA,
                                  d_forceparams[type].pdihs.phiB,
                                  d_forceparams[type].pdihs.mult,
                                  phi,
                                  lambda_restr,
                                  &vid,
                                  &dVdphi); 

        // printf("i: %d, cosine of phi: %f, phi: %f.\n", i, cos_phi, phi); 
        // printf("ai: %d, aj: %d, ak: %d, al: %d.\n", ai, aj, ak, al); 
        // printf("vid: %f, dvdllambda: %f.\n", vid, dvdlambda); 
        if (calcEner)
        {
            *vtot_loc += vid;
            *dvdltot_loc += dvdlambda;
        }

        float cos_phi2 = cos_phi * cos_phi;
        if (cos_phi2 < 1.0F)
        {
            float st    = -dVdphi * rsqrtf(1.0F - cos_phi2);
            float sth   = st * cos_phi;
            float nrij2 = norm2(r_ij);
            float nrkl2 = norm2(r_kl);

            float nrij_1 = rsqrtf(nrij2);
            float nrkl_1 = rsqrtf(nrkl2);

            float c = st * nrij_1 * nrkl_1;
            float cij = sth * nrij_1 * nrij_1;
            float ckl = sth * nrkl_1 * nrkl_1;

            float3 f_i = c * r_kl - cij * r_ij;
            float3 f_k = c * r_ij - ckl * r_kl;

            staggeredAtomicAddForce(&gm_f[ai], f_i);
            staggeredAtomicAddForce(&gm_f[aj], -f_i);
            staggeredAtomicAddForce(&gm_f[ak], f_k);
            staggeredAtomicAddForce(&gm_f[al], -f_k);

            if (calcVir)
            {
                staggeredAtomicAddForce(&sm_fShiftLoc[t1], f_i);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -f_i);
                staggeredAtomicAddForce(&sm_fShiftLoc[t2], f_k);
                staggeredAtomicAddForce(&sm_fShiftLoc[gmx::c_centralShiftIndex], -f_k);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ __forceinline__ void dihres_gpu(const int       i,
                                          float*          vtot_loc,
                                          const int       numBonds,
                                          const t_iatom   d_forceatoms[],
                                          const t_iparams d_forceparams[],
                                          const float     lambda,
                                          const float4    gm_xq[],
                                          float3          gm_f[],
                                          float3          sm_fShiftLoc[],
                                          const PbcAiuc   pbcAiuc,
                                          float*          dvdltot_loc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float L1 = 1.0F - lambda;
        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;

        float phi0A = d_forceparams[type].dihres.phiA * CUDA_DEG2RAD_F;
        float dphiA = d_forceparams[type].dihres.dphiA * CUDA_DEG2RAD_F;
        float kfacA = d_forceparams[type].dihres.kfacA;

        float phi0B = d_forceparams[type].dihres.phiB * CUDA_DEG2RAD_F;
        float dphiB = d_forceparams[type].dihres.dphiB * CUDA_DEG2RAD_F;
        float kfacB = d_forceparams[type].dihres.kfacB;

        float phi0 = L1 * phi0A + lambda * phi0B;
        float dphi = L1 * dphiA + lambda * dphiB;
        float kfac = L1 * kfacA + lambda * kfacB;

        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        float dp = phi - phi0;
        make_dp_periodic_gpu(&dp);

        float dvdlambda = 0.0F;
        float ddp;

        if (dp > dphi)
        {
            ddp = dp - dphi;
        }
        else if (dp < -dphi)
        {
            ddp = dp + dphi;
        }
        else
        {
            ddp = 0.0F;
        }


        float ddp2 = ddp * ddp;
        float vpd = 0.5F * kfac * ddp2;
        float ddphi = kfac * ddp;

        dvdlambda += 0.5F * (kfacB - kfacA) * ddp2;
        /* lambda dependence from changing restraint distances */
        if (ddp > 0.0F)
        {
            dvdlambda -= kfac * ddp * ((dphiB - dphiA) + (phi0B - phi0A));
        }
        else if (ddp < 0.0F)
        {
            dvdlambda += kfac * ddp * ((dphiB - dphiA) - (phi0B - phi0A));
        }

        if (calcEner)
        {
            *vtot_loc += vpd;
            *dvdltot_loc += dvdlambda;
        }

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
    }
}

namespace gmx
{

template<bool calcVir, bool calcEner>
__global__ void bonded_kernel_gpu(BondedGpuKernelParameters kernelParams,
                                  BondedGpuKernelBuffers    kernelBuffers,
                                  float4*                   gm_xq,
                                  float3*                   gm_f,
                                  float3*                   gm_fShift,
                                  float4*                   gm_q4)
{
    assert(blockDim.y == 1 && blockDim.z == 1);
    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    float     vtot_loc     = 0.0F;
    float     vtotElec_loc = 0.0F;
    float     dvdltot_loc = 0.0F;
    float     dvdltotElec_loc = 0.0F;

    extern __shared__ char sm_dynamicShmem[];
    char*                  sm_nextSlotPtr = sm_dynamicShmem;
    float3*                sm_fShiftLoc   = reinterpret_cast<float3*>(sm_nextSlotPtr);
    sm_nextSlotPtr += c_numShiftVectors * sizeof(float3);

    if (calcVir)
    {
        if (threadIdx.x < c_numShiftVectors)
        {
            sm_fShiftLoc[threadIdx.x] = make_float3(0.0F, 0.0F, 0.0F);
        }
        __syncthreads();
    }

    int  fType;
    bool threadComputedPotential = false;
    bool bFEP = false;
#pragma unroll
    for (int j = 0; j < numFTypesOnGpu; j++)
    {
        if (tid >= kernelParams.fTypeRangeStart[j] && tid <= kernelParams.fTypeRangeEnd[j])
        {
            const int      numBonds = kernelParams.numFTypeBonds[j];
            int            fTypeTid = tid - kernelParams.fTypeRangeStart[j];
            const t_iatom* iatoms   = kernelBuffers.d_iatoms[j];
            fType                   = kernelParams.fTypesOnGpu[j];
            if (calcEner)
            {
                threadComputedPotential = true;
            }
            if (gm_q4 != nullptr && fType != F_LJC14_Q && fType != F_LJC_PAIRS_NB) // bonded GPU FEP
            {
                bFEP = true;
                switch (fType)
                {
                    case F_BONDS:
                    case F_HARMONIC:
                        bonds_fep_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, kernelBuffers.d_fepParams, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_ANGLES:
                        angles_fep_gpu<calcVir, calcEner>(
                                fTypeTid, &vtot_loc, numBonds, iatoms, kernelBuffers.d_forceParams, kernelBuffers.d_fepParams,
                                gm_xq, gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_UREY_BRADLEY:
                        urey_bradley_fep_gpu<calcVir, calcEner>(
                                fTypeTid, &vtot_loc, numBonds, iatoms, kernelBuffers.d_forceParams, kernelBuffers.d_fepParams,
                                gm_xq, gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_PDIHS:
                    case F_PIDIHS:
                        pdihs_fep_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, kernelBuffers.d_fepParams, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_RBDIHS:
                        rbdihs_fep_gpu<calcVir, calcEner>(
                                fTypeTid, &vtot_loc, numBonds, iatoms, kernelBuffers.d_forceParams, kernelBuffers.d_fepParams,
                                gm_xq, gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_IDIHS:
                        idihs_fep_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, kernelBuffers.d_fepParams, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_LJ14:
                        pairs_fep_gpu<calcVir, calcEner>(fTypeTid, numBonds, iatoms, kernelBuffers.d_forceParams, kernelBuffers.d_fepParams,
                                                    gm_xq, gm_q4, gm_f, sm_fShiftLoc,
                                                    kernelParams.pbcAiuc, kernelParams.electrostaticsScaleFactor,
                                                    &vtot_loc, &vtotElec_loc, &dvdltot_loc, &dvdltotElec_loc);
                        break;
                    case F_RESTRBONDS:
                        restraint_bonds_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, kernelBuffers.d_fepParams->lambdaRestr, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_ANGRES:
                        angleres_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, kernelBuffers.d_fepParams->lambdaRestr, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_DIHRES:
                        dihres_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, kernelBuffers.d_fepParams->lambdaRestr, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                }
            } else {
                switch (fType)
                {
                    case F_BONDS:
                    case F_HARMONIC:
                        bonds_gpu<calcVir, calcEner>(fTypeTid,
                                                    &vtot_loc,
                                                    numBonds,
                                                    iatoms,
                                                    kernelBuffers.d_forceParams,
                                                    gm_xq,
                                                    gm_f,
                                                    sm_fShiftLoc,
                                                    kernelParams.pbcAiuc);
                        break;
                    case F_ANGLES:
                        angles_gpu<calcVir, calcEner>(fTypeTid,
                                                    &vtot_loc,
                                                    numBonds,
                                                    iatoms,
                                                    kernelBuffers.d_forceParams,
                                                    gm_xq,
                                                    gm_f,
                                                    sm_fShiftLoc,
                                                    kernelParams.pbcAiuc);
                        break;
                    case F_UREY_BRADLEY:
                        urey_bradley_gpu<calcVir, calcEner>(fTypeTid,
                                                            &vtot_loc,
                                                            numBonds,
                                                            iatoms,
                                                            kernelBuffers.d_forceParams,
                                                            gm_xq,
                                                            gm_f,
                                                            sm_fShiftLoc,
                                                            kernelParams.pbcAiuc);
                        break;
                    case F_PDIHS:
                    case F_PIDIHS:
                        pdihs_gpu<calcVir, calcEner>(fTypeTid,
                                                    &vtot_loc,
                                                    numBonds,
                                                    iatoms,
                                                    kernelBuffers.d_forceParams,
                                                    gm_xq,
                                                    gm_f,
                                                    sm_fShiftLoc,
                                                    kernelParams.pbcAiuc);
                        break;
                    case F_RBDIHS:
                        rbdihs_gpu<calcVir, calcEner>(fTypeTid,
                                                    &vtot_loc,
                                                    numBonds,
                                                    iatoms,
                                                    kernelBuffers.d_forceParams,
                                                    gm_xq,
                                                    gm_f,
                                                    sm_fShiftLoc,
                                                    kernelParams.pbcAiuc);
                        break;
                    case F_IDIHS:
                        idihs_gpu<calcVir, calcEner>(fTypeTid,
                                                    &vtot_loc,
                                                    numBonds,
                                                    iatoms,
                                                    kernelBuffers.d_forceParams,
                                                    gm_xq,
                                                    gm_f,
                                                    sm_fShiftLoc,
                                                    kernelParams.pbcAiuc);
                        break;
                    case F_LJ14:
                        pairs_gpu<calcVir, calcEner>(fTypeTid,
                                                    numBonds,
                                                    iatoms,
                                                    kernelBuffers.d_forceParams,
                                                    gm_xq,
                                                    gm_f,
                                                    sm_fShiftLoc,
                                                    kernelParams.pbcAiuc,
                                                    kernelParams.electrostaticsScaleFactor,
                                                    &vtot_loc,
                                                    &vtotElec_loc,
                                                    0);
                        break;
                    case F_LJC14_Q:
                        pairs_gpu<calcVir, calcEner>(fTypeTid,
                                                    numBonds,
                                                    iatoms,
                                                    kernelBuffers.d_forceParams,
                                                    gm_xq,
                                                    gm_f,
                                                    sm_fShiftLoc,
                                                    kernelParams.pbcAiuc,
                                                    kernelParams.epsFac,
                                                    &vtot_loc,
                                                    &vtotElec_loc,
                                                    1);
                        break;
                    case F_LJC_PAIRS_NB:
                        pairs_gpu<calcVir, calcEner>(fTypeTid,
                                                    numBonds,
                                                    iatoms,
                                                    kernelBuffers.d_forceParams,
                                                    gm_xq,
                                                    gm_f,
                                                    sm_fShiftLoc,
                                                    kernelParams.pbcAiuc,
                                                    kernelParams.epsFac,
                                                    &vtot_loc,
                                                    &vtotElec_loc,
                                                    2);
                        break;
                    case F_RESTRBONDS:
                        restraint_bonds_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, 1.0, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_ANGRES:
                        angleres_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, 1.0, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                    case F_DIHRES:
                        dihres_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                    kernelBuffers.d_forceParams, 1.0, gm_xq,
                                                    gm_f, sm_fShiftLoc, kernelParams.pbcAiuc, &dvdltot_loc);
                        break;
                }
            }
            break;
        }
    }

    if (threadComputedPotential)
    {
        float* vtot     = kernelBuffers.d_vTot + fType;
        float* vtotElec = kernelBuffers.d_vTot + F_COUL14;
        if (fType == F_LJC_PAIRS_NB)
            vtotElec = kernelBuffers.d_vTot + F_COUL_SR;

        // Perform warp-local reduction
        vtot_loc += __shfl_down_sync(c_fullWarpMask, vtot_loc, 1);
        vtotElec_loc += __shfl_up_sync(c_fullWarpMask, vtotElec_loc, 1);
        if (threadIdx.x & 1)
        {
            vtot_loc = vtotElec_loc;
        }
#pragma unroll 4
        for (int i = 2; i < warpSize; i *= 2)
        {
            vtot_loc += __shfl_down_sync(c_fullWarpMask, vtot_loc, i);
        }

        // Write reduced warp results into global memory
        if (threadIdx.x % warpSize == 0)
        {
            atomicAdd(vtot, vtot_loc);
        }
        else if ((threadIdx.x % warpSize == 1) && (fType == F_LJ14 || fType == F_LJC14_Q || fType == F_LJC_PAIRS_NB))
        {
            atomicAdd(vtotElec, vtot_loc);
        }
    }
    /* Accumulate shift vectors from shared memory to global memory on the first c_numShiftVectors threads of the block. */
    if (calcVir)
    {
        __syncthreads();
        if (threadIdx.x < c_numShiftVectors)
        {
            staggeredAtomicAddForce(&gm_fShift[threadIdx.x], sm_fShiftLoc[threadIdx.x]);
        }
    }

    if (bFEP)
    {
        float* dvdltot     = kernelBuffers.d_dvdlTot;
        if (fType == F_LJ14)
            // dvdlVdw
            dvdltot += 3;
        else if (fType == F_RESTRBONDS || fType == F_ANGRES || fType == F_DIHRES)
            // dvdlRestr
            dvdltot += 5;
        else
            // dvdlBonded
            dvdltot += 4;
        // dvdlElec, for F_LJ14 only
        float* dvdltotElec = kernelBuffers.d_dvdlTot + 2;

        // Perform warp-local reduction
        dvdltot_loc += __shfl_down_sync(c_fullWarpMask, dvdltot_loc, 1);
        dvdltotElec_loc += __shfl_up_sync(c_fullWarpMask, dvdltotElec_loc, 1);
        if (threadIdx.x & 1)
        {
            dvdltot_loc = dvdltotElec_loc;
        }
#pragma unroll 4
        for (int i = 2; i < warpSize; i *= 2)
        {
            dvdltot_loc += __shfl_down_sync(c_fullWarpMask, dvdltot_loc, i);
        }

        // Write reduced warp results into global memory
        if (threadIdx.x % warpSize == 0)
        {
            atomicAdd(dvdltot, dvdltot_loc);
        }
        else if ((threadIdx.x % warpSize == 1) && (fType == F_LJ14))
        {
            atomicAdd(dvdltotElec, dvdltot_loc);
        }
    }
}


/*-------------------------------- End CUDA kernels-----------------------------*/


template<bool calcVir, bool calcEner>
void ListedForcesGpu::Impl::launchKernel()
{
    GMX_ASSERT(haveInteractions_,
               "Cannot launch bonded GPU kernels unless bonded GPU work was scheduled");

    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuBonded);

    int fTypeRangeEnd = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];

    if (fTypeRangeEnd < 0)
    {
        return;
    }

    auto kernelPtr = bonded_kernel_gpu<calcVir, calcEner>;

    const auto kernelArgs = prepareGpuKernelArguments(
            kernelPtr, kernelLaunchConfig_, &kernelParams_, &kernelBuffers_, &d_xq_, &d_f_, &d_fShift_, &d_q4_);

    if (debug)
    {
        fprintf(debug,
                "Bonded GPU launch configuration:\n\tThread block: %zux%zux%zu\n\t"
                "\tGrid: %zux%zu\n",
                kernelLaunchConfig_.blockSize[0],
                kernelLaunchConfig_.blockSize[1],
                kernelLaunchConfig_.blockSize[2],
                kernelLaunchConfig_.gridSize[0],
                kernelLaunchConfig_.gridSize[1]
                );
    }

    launchGpuKernel(kernelPtr,
                    kernelLaunchConfig_,
                    deviceStream_,
                    nullptr,
                    "bonded_kernel_gpu<calcVir, calcEner>",
                    kernelArgs);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpuPp);
}

void ListedForcesGpu::launchKernel(const gmx::StepWorkload& stepWork)
{
    if (stepWork.computeEnergy)
    {
        // When we need the energy, we also need the virial
        impl_->launchKernel<true, true>();
    }
    else if (stepWork.computeVirial)
    {
        impl_->launchKernel<true, false>();
    }
    else
    {
        impl_->launchKernel<false, false>();
    }
}

} // namespace gmx
