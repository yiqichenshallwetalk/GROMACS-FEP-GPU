/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016,2017,2018,2019, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
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
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/*! \internal \file
 *  \brief
 *  CUDA FEP non-bonded kernel used through preprocessor-based code generation
 *  of multiple kernel flavors, see nbnxn_fep_cuda_kernels.cuh.
 *
 *  NOTE: No include fence as it is meant to be included multiple times.
 *
 */

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/cuda_kernel_utils.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/pbcutil/ishift.h"
/* Note that floating-point constants in CUDA code should be suffixed
 * with f (e.g. 0.5f), to stop the compiler producing intermediate
 * code that is in double precision.
 */

#if defined EL_EWALD_ANA || defined EL_EWALD_TAB
/* Note: convenience macro, needs to be undef-ed at the end of the file. */
#    define EL_EWALD_ANY
#endif

#if defined EL_EWALD_ANY || defined EL_RF || defined LJ_EWALD \
        || (defined EL_CUTOFF && defined CALC_ENERGIES)
/* Macro to control the calculation of exclusion forces in the kernel
 * We do that with Ewald (elec/vdw) and RF. Cut-off only has exclusion
 * energy terms.
 *
 * Note: convenience macro, needs to be undef-ed at the end of the file.
 */
#    define EXCLUSION_FORCES
#endif

#if defined LJ_EWALD_COMB_GEOM || defined LJ_EWALD_COMB_LB
/* Note: convenience macro, needs to be undef-ed at the end of the file. */
#    define LJ_EWALD
#endif

#if defined LJ_COMB_GEOM || defined LJ_COMB_LB
#    define LJ_COMB
#endif

/*
    Each warp is in charge of one entry in nri.

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */

#ifdef CALC_ENERGIES
        __global__ void NB_FEP_KERNEL_FUNC_NAME(nbnxn_fep_kernel, _VF_cuda)
#else
        __global__ void NB_FEP_KERNEL_FUNC_NAME(nbnxn_fep_kernel, _F_cuda)
#endif  /* CALC_ENERGIES */
        (const NBAtomDataGpu atdat,
         const NBParamGpu  nbparam,
         const Nbnxm::gpu_feplist  feplist,
         bool bCalcFshift)
#ifdef FUNCTION_DECLARATION_ONLY
                ; /* Only do function declaration, omit the function body. */
#else
{
    /* convenience variables */
    // constexpr float minDistanceSquared = 1.0e-12F;
    const float alphaCoulomb     = nbparam.alpha_coul;
    const float alphaVdw      = nbparam.alpha_vdw;
    float       alphaCoulombEff = alphaCoulomb;
    float       alphaVdwEff  = alphaVdw;
    const bool  useSoftCore    = (alphaVdw != 0.0F);
    const float sigma6_def     = nbparam.sc_sigma6;
    const float sigma6_min     = nbparam.sc_sigma6_min;
    const float lambdaCoul       = nbparam.lambda_q;
    const float _lambdaCoul      = 1.0F - lambdaCoul;
    const float lambdaVdw       = nbparam.lambda_v;
    const float _lambdaVdw      = 1.0F - lambdaVdw;

    float lambdaFactorCoul[2]       = { _lambdaCoul, lambdaCoul };
    float lambdaFactorVdw[2]       = { _lambdaVdw, lambdaVdw };
    float softcoreLambdaFactorCoul[2] = { lambdaCoul, _lambdaCoul };
    float softcoreLambdaFactorVdw[2]  = { lambdaVdw, _lambdaVdw };

#    ifndef LJ_COMB
    const int4* atomTypes4 = atdat.atomTypes4;
    int        numTypes      = atdat.numTypes;
    int        typeiAB[2], typejAB[2];
#    else
    const float4* ljComb4 = atdat.ljComb4;
#    endif

    float rInvC, r2C, rPInvC, rPInvV, rInv6;
#    if defined LJ_POT_SWITCH
    float rInvV, r2V, rV;
#    endif
    float sigma6[2], c6AB[2], c12AB[2];
    float qq[2];
    float scalarForcePerDistanceCoul[2], scalarForcePerDistanceVdw[2];

#    ifdef CALC_ENERGIES
    float Vcoul[2];
#    endif
#    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
    float Vvdw[2];
#    endif

#    ifdef LJ_COMB_LB
    float sigmaAB[2];
    float epsilonAB[2];
#    endif

    const float4* xq          = atdat.xq;
    const float4* q4          = atdat.q4;
    float3*       f           = asFloat3(atdat.f);
    const float3* shiftVec   = asFloat3(atdat.shiftVec);

    float         rCutoffCoulSq = nbparam.rcoulomb_sq;
    float         rCutoffMaxSq = rCutoffCoulSq;
#    ifdef VDW_CUTOFF_CHECK
    float         rCutoffVdwSq     = nbparam.rvdw_sq;
    float         vdw_in_range;
    rCutoffMaxSq = max(rCutoffCoulSq, rCutoffVdwSq);
#    endif
#    ifdef EL_RF
    float         two_k_rf = nbparam.two_k_rf;
#    endif
#    ifdef EL_EWALD_ANA
    float         beta2    = nbparam.ewald_beta * nbparam.ewald_beta;
    float         beta3    = nbparam.ewald_beta * nbparam.ewald_beta * nbparam.ewald_beta;
#    endif

#    ifdef EL_EWALD_ANY
    float         beta     = nbparam.ewald_beta;
    float         v_lr, f_lr;
#    endif

#    ifdef CALC_ENERGIES
#        ifdef EL_EWALD_ANY
    float         ewald_shift = nbparam.sh_ewald;
#        else
    float c_rf = nbparam.c_rf;
#        endif /* EL_EWALD_ANY */

    float*        e_lj        = atdat.eLJ;
    float*        e_el        = atdat.eElec;
    float*        dvdl_lj     = atdat.dvdlLJ;
    float*        dvdl_el     = atdat.dvdlElec;
    const int     lambdaPower   = nbparam.lam_power;

    float dLambdaFactor[2];
    float softcoreDlFactorCoul[2];
    float softcoreDlFactorVdw[2];

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
#    endif     /* CALC_ENERGIES */

    /* thread/block/warp id-s */
    int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    int tid_global = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
                            + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
                            + threadIdx.x;
    int wid_global = tid_global / warp_size;
    // thread Id within a warp
    int tid_in_warp = tid % warp_size;

#    ifndef LJ_COMB
#    else
    float2        ljcp_iAB[2], ljcp_jAB[2];
#    endif
    float qAi, qAj_f, r2, rpm2, rp, inv_r, inv_r2;
    float qBi, qBj_f;

    float scalarForcePerDistance;
#    ifdef CALC_ENERGIES
    float  E_lj, E_el;
    float  DVDL_lj, DVDL_el;
#    endif

    float4 xqbuf, q4_buf;
#    ifndef LJ_COMB
    int4 atomTypes4_buf;
#    else
    float4 ljComb4_buf;
#    endif

    float3 xi, xj, rv, f_ij, fci, fcj;

    // Extract pair list data
    const int  nri    = feplist.nri;
    const int* iinr   = feplist.iinr;
    const int* jindex = feplist.jindex;
    const int* jjnr   = feplist.jjnr;
    const int* shift  = feplist.shift;

#    ifdef CALC_ENERGIES
    E_lj         = 0.0F;
    E_el         = 0.0F;
    DVDL_lj      = 0.0F;
    DVDL_el      = 0.0F;
#    endif /* CALC_ENERGIES */

    // One warp for one ri entry
    if (wid_global < nri)
    {
        const int nj0      = __shfl_sync(c_fullWarpMask, jindex[wid_global], 0, warp_size);
        const int nj1      = __shfl_sync(c_fullWarpMask, jindex[wid_global + 1], 0, warp_size);
        const int ai      = __shfl_sync(c_fullWarpMask, iinr[wid_global], 0, warp_size);
        const float3    shiftI = __shfl_sync(c_fullWarpMask, shiftVec[shift[wid_global]], 0, warp_size);

        xqbuf = __shfl_sync(c_fullWarpMask, xq[ai], 0, warp_size);
        xqbuf = xqbuf + make_float4(shiftI.x, shiftI.y, shiftI.z, 0.0F);
        xi  = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);

        q4_buf = q4[ai];
        qAi = q4_buf.x * nbparam.epsfac;
        qBi = q4_buf.y * nbparam.epsfac;

#    ifndef LJ_COMB
        atomTypes4_buf = atomTypes4[ai];
        typeiAB[0] = atomTypes4_buf.x;
        typeiAB[1] = atomTypes4_buf.y;

#    else
        ljComb4_buf = ljComb4[ai];
        ljcp_iAB[0] = make_float2(ljComb4_buf.x, ljComb4_buf.y);
        ljcp_iAB[1] = make_float2(ljComb4_buf.z, ljComb4_buf.w);
#    endif
        fci = make_float3(0.0F);
#    pragma unroll
        for (int i = nj0; i < nj1; i += warp_size)
        {
            int j = i + tid_in_warp;
            if (j < nj1)
            {
                const int aj = jjnr[j];
                bool pairIncluded = (feplist.excl_fep == nullptr || feplist.excl_fep[j]);
                scalarForcePerDistance = 0.0F; // F_invr
                scalarForcePerDistanceCoul[0] = scalarForcePerDistanceCoul[1] = 0.0F;
                scalarForcePerDistanceVdw[0] = scalarForcePerDistanceVdw[1] = 0.0F;
    #    ifdef CALC_ENERGIES
                Vcoul[0] = Vcoul[1] = 0.0F;
    #    endif
    #    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                Vvdw[0] = Vvdw[1] = 0.0F;
    #    endif
                /* load j atom data */
                xqbuf = xq[aj];
                xj    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
                q4_buf = q4[aj];
                qAj_f = q4_buf.x;
                qBj_f = q4_buf.y;
                qq[0] = qAi * qAj_f;
                qq[1] = qBi * qBj_f;
    #    ifndef LJ_COMB
                atomTypes4_buf = atomTypes4[aj];
                typejAB[0] = atomTypes4_buf.x;
                typejAB[1] = atomTypes4_buf.y;

    #    else
                ljComb4_buf = ljComb4[aj];
                ljcp_jAB[0] = make_float2(ljComb4_buf.x, ljComb4_buf.y);
                ljcp_jAB[1] = make_float2(ljComb4_buf.z, ljComb4_buf.w);
    #    endif
                fcj    = make_float3(0.0F);
                /* distance between i and j atoms */
                rv = xi - xj;
                r2 = norm2(rv);
                bool withinCutoffMask = (r2 < rCutoffMaxSq);
                if (!withinCutoffMask && pairIncluded)
                    continue;
                // Ensure distance do not become so small that r^-12 overflows
                r2     = max(r2, c_nbnxnMinDistanceSquared);
                inv_r = rsqrt(r2);
                inv_r2 = inv_r * inv_r;

                if (pairIncluded)
                {
                    if (useSoftCore)
                    {
                        rpm2 = r2 * r2;
                        rp   = rpm2 * r2;
                    }
                    else
                    {
                        rpm2 = inv_r * inv_r;
                        rp   = 1.0F;
                    }

                    for (int k = 0; k < 2; k++)
                    {
    #    ifndef LJ_COMB
                        /* LJ 6*C6 and 12*C12 */
                        fetch_nbfp_c6_c12(c6AB[k], c12AB[k], nbparam,
                                            numTypes * typeiAB[k] + typejAB[k]);
                        if (useSoftCore)
                            convert_c6_c12_to_sigma6(c6AB[k], c12AB[k], &(sigma6[k]), sigma6_min, sigma6_def);
    #    else
    #        ifdef LJ_COMB_GEOM
                        c6AB[k]  = ljcp_iAB[k].x * ljcp_jAB[k].x;
                        c12AB[k] = ljcp_iAB[k].y * ljcp_jAB[k].y;
                        if (useSoftCore)
                            convert_c6_c12_to_sigma6(c6AB[k], c12AB[k], &(sigma6[k]), sigma6_min, sigma6_def);
    #        else
                        /* LJ 2^(1/6)*sigma and 12*epsilon */
                        sigmaAB[k]   = ljcp_iAB[k].x + ljcp_jAB[k].x;
                        if (ljcp_iAB[k].x == 0.0F || ljcp_jAB[k].x == 0.0F)
                            sigmaAB[k] = 0.0F;
                        epsilonAB[k] = ljcp_iAB[k].y * ljcp_jAB[k].y;
                        convert_sigma_epsilon_to_c6_c12(sigmaAB[k], epsilonAB[k], &(c6AB[k]),
                                                        &(c12AB[k]));
                        if (useSoftCore)
                        {
                            if ((c6AB[k] > 0.0F) && (c12AB[k] > 0.0F)) {
                                float sigma2 = sigmaAB[k] * sigmaAB[k];
                                sigma6[k]    = sigma2 * sigma2 * sigma2 * 0.5F;
                                if (sigma6[k] < sigma6_min)
                                    sigma6[k] = sigma6_min;
                            }
                            else {
                                sigma6[k] = sigma6_def;
                            }
                        }

    #        endif /* LJ_COMB_GEOM */
    #    endif     /* LJ_COMB */
                    }
                    if (useSoftCore)
                    {
                        /* only use softcore if one of the states has a zero endstate - softcore is for avoiding infinities!*/
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
    #    ifdef CALC_ENERGIES
                        Vcoul[k]  = 0.0F;
    #    endif
    #    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                        Vvdw[k]   = 0.0F;
    #    endif

                        bool nonZeroState = ((qq[k] != 0.0F) || (c6AB[k] != 0.0F) || (c12AB[k] != 0.0F));
                        if (nonZeroState)
                        {
                            if (useSoftCore)
                            {
                                rPInvC = 1.0F / (alphaCoulombEff * softcoreLambdaFactorCoul[k] * sigma6[k] + rp);
                                r2C   = rcbrt(rPInvC);
                                rInvC = rsqrt(r2C);

                                // equivalent to scLambdasOrAlphasDiffer
                                if ((alphaCoulombEff != alphaVdwEff) || (softcoreLambdaFactorVdw[k] != softcoreLambdaFactorCoul[k]))
                                {
                                    rPInvV = 1.0F / (alphaVdwEff * softcoreLambdaFactorVdw[k] * sigma6[k] + rp);
#    if defined LJ_POT_SWITCH
                                    r2V    = rcbrt(rPInvV);
                                    rInvV = rsqrt(r2V);
                                    rV     = r2V * rInvV;
#    endif
                                }
                                else
                                {
                                    /* We can avoid one expensive pow and one / operation */
                                    rPInvV = rPInvC;
#    if defined LJ_POT_SWITCH
                                    r2V    = r2C;
                                    rInvV  = rInvC;
                                    rV     = r2V * rInvV;
#    endif
                                }
                            }
                            else
                            {
                                rPInvC = 1.0F;
                                r2C    = r2;
                                rInvC  = inv_r;

                                rPInvV = 1.0F;
#    if defined LJ_POT_SWITCH
                                r2V    = r2;
                                rInvV  = inv_r;
                                rV     = r2V * rInvV;
#    endif
                            }

                            if (c6AB[k] != 0.0F || c12AB[k] != 0.0F)
                            {
                                if (!useSoftCore)
                                {
                                    rInv6 = inv_r2 * inv_r2 * inv_r2;
                                }
                                else
                                {
                                    rInv6 = rPInvV;
                                }
                                float Vvdw6                     = c6AB[k] * rInv6;
                                float Vvdw12                    = c12AB[k] * rInv6 * rInv6;
                                scalarForcePerDistanceVdw[k]    = Vvdw12 - Vvdw6;
    #    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                                Vvdw[k]      =
                                        ((Vvdw12 + c12AB[k] * nbparam.repulsion_shift.cpot) * c_oneTwelveth
                                            - (Vvdw6 + c6AB[k] * nbparam.dispersion_shift.cpot)
                                                    * c_oneSixth);
    #    endif

    #    ifdef LJ_POT_SWITCH
    #        ifdef CALC_ENERGIES
                                calculate_potential_switch_Fr_E(
                                        nbparam, rV, &(scalarForcePerDistanceVdw[k]), &(Vvdw[k]));
#        else
                                calculate_potential_switch_Fr(
                                        nbparam, rV, &(scalarForcePerDistanceVdw[k]), &(Vvdw[k]));
#        endif /* CALC_ENERGIES */
    #    endif     /* LJ_POT_SWITCH */

    #    ifdef VDW_CUTOFF_CHECK
                                /* Separate VDW cut-off check to enable twin-range cut-offs
                                * (rvdw < rcoulomb <= rlist)
                                */
                                vdw_in_range = (r2 < rCutoffVdwSq) ? 1.0F : 0.0F;
                                scalarForcePerDistanceVdw[k] *= vdw_in_range;
    #       ifdef CALC_ENERGIES
                                Vvdw[k] *= vdw_in_range;
    #       endif
    #    endif /* VDW_CUTOFF_CHECK */
                            }

                            if (qq[k] != 0.0F)
                            {
    #    ifdef EL_CUTOFF
    #        ifdef EXCLUSION_FORCES
                                scalarForcePerDistanceCoul[k] = qq[k] * rInvC;
    #        else
                                scalarForcePerDistanceCoul[k] = qq[k] * rInvC;
    #        endif
    #    endif
    #    ifdef EL_RF
                                scalarForcePerDistanceCoul[k] = qq[k] * (rInvC - two_k_rf * r2C);
    #    endif
    #    if defined EL_EWALD_ANY
                                scalarForcePerDistanceCoul[k] = qq[k] * rInvC;
    #    endif /* EL_EWALD_ANA/TAB */

    #    ifdef CALC_ENERGIES
    #        ifdef EL_CUTOFF
                                Vcoul[k]  = qq[k] * (rInvC - c_rf);
    #        endif
    #        ifdef EL_RF
                                Vcoul[k] = qq[k] * (rInvC + 0.5F * two_k_rf * r2C - c_rf);
    #        endif
    #        ifdef EL_EWALD_ANY
                                /* 1.0f - erff is faster than erfcf */
                                Vcoul[k] = qq[k] * (rInvC - ewald_shift);
    #        endif /* EL_EWALD_ANY */
    #    endif
                            }
                            scalarForcePerDistanceCoul[k] *= rPInvC;
                            scalarForcePerDistanceVdw[k] *= rPInvV;
                        }
                    }// end for (int k = 0; k < 2; k++)

                    for (int k = 0; k < 2; k++)
                    {
#    ifdef CALC_ENERGIES
                        E_el = E_el + lambdaFactorCoul[k] * Vcoul[k];
                        E_lj = E_lj + lambdaFactorVdw[k] * Vvdw[k];

                        if (useSoftCore)
                        {
                            DVDL_el = DVDL_el + Vcoul[k] * dLambdaFactor[k]
                                        + lambdaFactorCoul[k] * alphaCoulombEff * softcoreDlFactorCoul[k] * scalarForcePerDistanceCoul[k] * sigma6[k];
                            DVDL_lj = DVDL_lj + Vvdw[k] * dLambdaFactor[k]
                                        + lambdaFactorVdw[k] * alphaVdwEff * softcoreDlFactorVdw[k] * scalarForcePerDistanceVdw[k] * sigma6[k];
                        }
                        else
                        {
                            DVDL_el = DVDL_el + Vcoul[k] * dLambdaFactor[k];
                            DVDL_lj = DVDL_lj + Vvdw[k] * dLambdaFactor[k];
                        }
#    endif
                        scalarForcePerDistance = scalarForcePerDistance + lambdaFactorCoul[k] * scalarForcePerDistanceCoul[k] * rpm2;
                        scalarForcePerDistance = scalarForcePerDistance + lambdaFactorVdw[k] * scalarForcePerDistanceVdw[k] * rpm2;
                    }
                } //if (pairIncluded)

        // ELEC REACTIONFIELD part
        #   if defined EL_CUTOFF || defined EL_RF
            if (!pairIncluded) {
                # if defined EL_CUTOFF
                float FF = 0.0F;
                float VV = -nbparam.c_rf;
                # else
                float FF = -two_k_rf;
                float VV = 0.5F * two_k_rf * r2 - nbparam.c_rf;
                # endif

                if (ai == aj)
                    VV *= 0.5F;
                for (int k = 0; k < 2; k++)
                    {
        #       ifdef CALC_ENERGIES
                        E_el += lambdaFactorCoul[k] * qq[k] * VV;
                        DVDL_el += dLambdaFactor[k] * qq[k] * VV;
        #       endif
                        scalarForcePerDistance += lambdaFactorCoul[k] * qq[k] * FF;
                    }
                }
        #   endif

    #    ifdef EL_EWALD_ANY
                if (!pairIncluded || r2 < rCutoffCoulSq)
                {
                    v_lr = inv_r > 0.0F ? inv_r * erff(r2 * inv_r * beta) : 2.0F * beta * M_FLOAT_1_SQRTPI;
                    if (ai == aj)
                        v_lr *= 0.5F;
    #        if defined   EL_EWALD_ANA
                    f_lr = inv_r > 0.0F ? -pmecorrF(beta2 * r2) * beta3 : 0.0F;
    #        elif defined EL_EWALD_TAB
                    f_lr = inv_r > 0.0F ? interpolate_coulomb_force_r(nbparam, r2 * inv_r) * inv_r: 0.0F;
    #        endif

                    for (int k = 0; k < 2; k++)
                    {
    #        ifdef CALC_ENERGIES
                        E_el -= lambdaFactorCoul[k] * qq[k] * v_lr;
                        DVDL_el -= dLambdaFactor[k] * qq[k] * v_lr;
    #        endif
                        scalarForcePerDistance -= lambdaFactorCoul[k] * qq[k] * f_lr;
                    }
                }
    #    endif

                if (scalarForcePerDistance != 0.0F) {
                    f_ij = rv * scalarForcePerDistance;

                    /* accumulate j forces in registers */
                    fcj -= f_ij;

                    /* accumulate i forces in registers */
                    fci += f_ij;
                    /* reduce j forces */
                    staggeredAtomicAddForce(&(f[aj]), fcj, tid);
                }
            } // end if (j < nj1)
        } // end for (int i = nj0; i < nj1; i += warp_size)
        reduce_fep_force_i_warp_shfl(fci, f, tid, ai, c_fullWarpMask);

        if (shift[wid_global] == gmx::c_centralShiftIndex)
        {
            bCalcFshift = false;
        }

        if (bCalcFshift)
        {
            float3* fShift = asFloat3(atdat.fShift);
            staggeredAtomicAddForce(&(fShift[shift[wid_global]]), fci, tid);
        }
    //#   endif

#    ifdef CALC_ENERGIES
        reduce_fep_energy_warp_shfl(E_lj, E_el, DVDL_lj, DVDL_el, e_lj, e_el, dvdl_lj, dvdl_el, tid, c_fullWarpMask);
#    endif
    } // end if (wid_global < nri)
}
#endif /* FUNCTION_DECLARATION_ONLY */

#undef EL_EWALD_ANY
#undef EXCLUSION_FORCES
#undef LJ_EWALD

#undef LJ_COMB
