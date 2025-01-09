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
 *  CUDA FEP foreign energy non-bonded kernel used through preprocessor-based code generation
 *  of multiple kernel flavors, see nbnxn_foreign_fep_cuda_kernels.cuh.
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


__global__ void NB_FOREIGN_FEP_KERNEL_FUNC_NAME(nbnxn_foreign_fep_kernel, _V_cuda)
      (     const NBAtomDataGpu atdat,
            const NBParamGpu  nbparam,
            const Nbnxm::gpu_feplist  feplist,
            const int n_lambda
      )
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

      #    ifndef LJ_COMB
      const int4* atomTypes4 = atdat.atomTypes4;
      int        numTypes      = atdat.numTypes;
      int        typeiAB[2], typejAB[2];
      #    else
      const float4* ljComb4 = atdat.ljComb4;
      #    endif

      float rInvC, r2C, rPInvC, rPInvV;
#    if defined LJ_POT_SWITCH
      float rInvV, r2V;
#    endif
      float sigma6[2], c6AB[2], c12AB[2];
      float qq[2];
      float scalarForcePerDistanceVdw[2];

      float Vcoul[2];
      float Vvdw[2];

#    ifdef LJ_COMB_LB
      float sigmaAB[2];
      float epsilonAB[2];
#    endif

      const float4* xq          = atdat.xq;
      const float4* q4          = atdat.q4;
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
      // #    ifdef EL_EWALD_ANA
      // float         beta2    = nbparam.ewald_beta * nbparam.ewald_beta;
      // float         beta3    = nbparam.ewald_beta * nbparam.ewald_beta * nbparam.ewald_beta;
      // #    endif

#    ifdef EL_EWALD_ANY
      float         beta     = nbparam.ewald_beta;
      float         v_lr;
#    endif

      //#    ifdef CALC_ENERGIES
#    ifdef EL_EWALD_ANY
      float         ewald_shift = nbparam.sh_ewald;
#    else
      float c_rf = nbparam.c_rf;
#    endif /* EL_EWALD_ANY */

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
      // float c6, c12;

      //float  int_bit;
      float  E_lj, E_el;
      float  DVDL_lj, DVDL_el;

      float4 xqbuf, q4_buf;
#    ifndef LJ_COMB
          int4 atomTypes4_buf;
#    else
          float4 ljComb4_buf;
#    endif

      float3 xi, xj, rv;

      // Extract pair list data
      const int  nri    = feplist.nri;
      const int* iinr   = feplist.iinr;
      const int* jindex = feplist.jindex;
      const int* jjnr   = feplist.jjnr;
      const int* shift  = feplist.shift;

      const int     lambdaPower   = nbparam.lam_power;
      constexpr float softcoreRPower = 6.0F;

      float dLambdaFactor[2];
      // float softcoreDlFactorCoul[2];
      float softcoreDlFactorVdw[2];

      /*derivative of the lambda factor for state A and B */
      dLambdaFactor[0] = -1.0F;
      dLambdaFactor[1] = 1.0F;

      const float* allLambdaCoul = nbparam.allLambdaCoul;
      const float* allLambdaVdw = nbparam.allLambdaVdw;
      float lambdaCoul, _lambdaCoul, lambdaVdw, _lambdaVdw;
      float lambdaFactorCoul[2], lambdaFactorVdw[2], softcoreLambdaFactorCoul[2], softcoreLambdaFactorVdw[2];

      extern __shared__ float sm_dynamicShmem[];
      float* lambdaCoulShmem = sm_dynamicShmem;
      float* lambdaVdwShmem = lambdaCoulShmem + n_lambda + 1;

      if (tid == 0) {
            lambdaCoulShmem[0] = nbparam.lambda_q;
            lambdaVdwShmem[0] = nbparam.lambda_v;
      }
      else if (tid <= n_lambda) {
            lambdaCoulShmem[tid] = allLambdaCoul[tid-1];
            lambdaVdwShmem[tid] = allLambdaVdw[tid-1];
      }

      __syncthreads();

      float* e_lj        = atdat.eLJForeign;
      float* e_el        = atdat.eElecForeign;
      float* dvdl_lj     = atdat.dvdlLJForeign;
      float* dvdl_el     = atdat.dvdlElecForeign;

      // Each warp calculates one ri
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

            int aj;
            bool pairIncluded;

            for (int i = nj0; i < nj1; i += warp_size)
            {
                  int j = i + tid_in_warp;
                  aj = jjnr[j];
                  pairIncluded = (feplist.excl_fep == nullptr || feplist.excl_fep[j] && j < nj1);
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

                  /* distance between i and j atoms */
                  rv = xi - xj;
                  r2 = norm2(rv);

                  bool withinCutoffMask = (r2 < rCutoffMaxSq);
                  // Ensure distance do not become so small that r^-12 overflows
                  r2     = max(r2, c_nbnxnMinDistanceSquared);
                  inv_r = rsqrt(r2);
                  inv_r2 = inv_r * inv_r;

                  for (int lambdaIdx = 0; lambdaIdx <= n_lambda; lambdaIdx ++)
                  {
                        E_lj = 0.0F;
                        E_el = 0.0F;
                        DVDL_lj = 0.0F;
                        DVDL_el = 0.0F;

                        lambdaCoul = lambdaCoulShmem[lambdaIdx];
                        lambdaVdw = lambdaVdwShmem[lambdaIdx];
                        _lambdaCoul = 1.0F - lambdaCoul;
                        _lambdaVdw = 1.0F - lambdaVdw;

                        lambdaFactorCoul[0]       = _lambdaCoul;
                        lambdaFactorCoul[1]       = lambdaCoul;
                        lambdaFactorVdw[0]       = _lambdaVdw;
                        lambdaFactorVdw[1]       = lambdaVdw;
                        softcoreLambdaFactorCoul[0] = lambdaCoul;
                        softcoreLambdaFactorCoul[1] = _lambdaCoul;
                        softcoreLambdaFactorVdw[0] = lambdaVdw;
                        softcoreLambdaFactorVdw[1] = _lambdaVdw;

                        for (int k = 0; k < 2; k++)
                        {
                              softcoreLambdaFactorCoul[k] =
                                    (lambdaPower == 2 ? (1.0F - lambdaFactorCoul[k]) * (1.0F - lambdaFactorCoul[k])
                                                      : (1.0F - lambdaFactorCoul[k]));
                              // softcoreDlFactorCoul[k] = dLambdaFactor[k] * lambdaPower / softcoreRPower
                              //                         * (lambdaPower == 2 ? (1.0F - lambdaFactorCoul[k]) : 1.0F);
                              softcoreLambdaFactorVdw[k] =
                                    (lambdaPower == 2 ? (1.0F - lambdaFactorVdw[k]) * (1.0F - lambdaFactorVdw[k])
                                                      : (1.0F - lambdaFactorVdw[k]));
                              softcoreDlFactorVdw[k] = dLambdaFactor[k] * lambdaPower / softcoreRPower
                                                      * (lambdaPower == 2 ? (1.0F - lambdaFactorVdw[k]) : 1.0F);
                        }

                        scalarForcePerDistanceVdw[0] = scalarForcePerDistanceVdw[1] = 0.0F;

                        if (pairIncluded && withinCutoffMask)
                        {
                              rpm2 = r2 * r2;
                              rp   = rpm2 * r2;

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
                                    scalarForcePerDistanceVdw[k] = 0.0F;
                                    Vcoul[k]  = 0.0F;
                                    Vvdw[k]   = 0.0F;

                                    bool nonZeroState = ((qq[k] != 0.0F) || (c6AB[k] != 0.0F) || (c12AB[k] != 0.0F));
                                    if (nonZeroState){
                                          if (useSoftCore)
                                          {
                                                rPInvC = 1.0F / (alphaCoulombEff * softcoreLambdaFactorCoul[k] * sigma6[k] + rp);
                                                r2C   = rcbrt(rPInvC);
                                                rInvC = rsqrt(r2C);

                                                // equivalent to scLambdasOrAlphasDiffer
                                                if ((alphaCoulombEff != alphaVdwEff) || (softcoreLambdaFactorVdw[k] != softcoreLambdaFactorCoul[k])) // || (alphaCoulomb != 0))
                                                {
                                                      rPInvV = 1.0F / (alphaVdwEff * softcoreLambdaFactorVdw[k] * sigma6[k] + rp);
#    if defined LJ_POT_SWITCH
                                                      r2V    = rcbrt(rPInvV);
                                                      rInvV = rsqrt(r2V);
#    endif
                                                }
                                                else
                                                {
                                                      /* We can avoid one expensive pow and one / operation */
                                                      rPInvV = rPInvC;
#    if defined LJ_POT_SWITCH
                                                      r2V    = r2C;
                                                      rInvV  = rInvC;
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
#    endif
                                          }

                                          if (c6AB[k] != 0.0F || c12AB[k] != 0.0F)
                                          {
                                                if (!useSoftCore) {
                                                      rPInvV = inv_r2 * inv_r2 * inv_r2;
                                                }
                                                float Vvdw6  = c6AB[k] * rPInvV;
                                                float Vvdw12 = c12AB[k] * rPInvV * rPInvV;
                                                scalarForcePerDistanceVdw[k]    = Vvdw12 - Vvdw6;

                                                Vvdw[k]      =
                                                            ((Vvdw12 + c12AB[k] * nbparam.repulsion_shift.cpot) * c_oneTwelveth
                                                            - (Vvdw6 + c6AB[k] * nbparam.dispersion_shift.cpot)
                                                                  * c_oneSixth);

#    ifdef LJ_POT_SWITCH
                                                calculate_potential_switch_F_E(nbparam, rInvV, r2V,
                                                                              &(scalarForcePerDistanceVdw[k]), &(Vvdw[k]));
#    endif /* LJ_POT_SWITCH */

#    ifdef VDW_CUTOFF_CHECK
                                                /* Separate VDW cut-off check to enable twin-range cut-offs
                                                * (rvdw < rcoulomb <= rlist)
                                                */
                                                vdw_in_range = (r2 < rCutoffVdwSq) ? 1.0F : 0.0F;
                                                Vvdw[k] *= vdw_in_range;
#    endif /* VDW_CUTOFF_CHECK */
                                          }

                                          if (qq[k] != 0.0F)
                                          {

#    ifdef EL_CUTOFF
                                                Vcoul[k]  = qq[k] * (rInvC - c_rf);
#    endif
#    ifdef EL_RF
                                                Vcoul[k] = qq[k] * (rInvC + 0.5F * two_k_rf * r2C - c_rf);
#    endif
#    ifdef EL_EWALD_ANY
                                                /* 1.0f - erff is faster than erfcf */
                                                Vcoul[k] = qq[k] * (rInvC - ewald_shift);
#    endif /* EL_EWALD_ANY */
                                          }
                                    }
                              }// end for (int k = 0; k < 2; k++)

                              for (int k = 0; k < 2; k++)
                              {
                                    E_el += lambdaFactorCoul[k] * Vcoul[k];
                                    E_lj += lambdaFactorVdw[k] * Vvdw[k];

                                    if (useSoftCore)
                                    {
                                          DVDL_el += Vcoul[k] * dLambdaFactor[k];
                                          //+ lambdaFactorCoul[k] * alphaCoulombEff * softcoreDlFactorCoul[k] * scalarForcePerDistanceCoul[k] * sigma6[k];
                                          DVDL_lj += Vvdw[k] * dLambdaFactor[k]
                                                      + lambdaFactorVdw[k] * alphaVdwEff * softcoreDlFactorVdw[k] * scalarForcePerDistanceVdw[k] * sigma6[k];
                                    }
                                    else
                                    {
                                          DVDL_el += Vcoul[k] * dLambdaFactor[k];
                                          DVDL_lj += Vvdw[k] * dLambdaFactor[k];
                                    }
                              }
                        } //if (pairIncluded && withinCutoffMask)

                        // ELEC REACTIONFIELD part
#    if defined EL_CUTOFF || defined EL_RF
                        if (!pairIncluded && j < nj1) {
#        if defined EL_CUTOFF
                              float VV = -nbparam.c_rf;
#        else
                              float VV = 0.5F * two_k_rf * r2 - nbparam.c_rf;
#        endif

                              if (ai == aj)
                                    VV *= 0.5F;
                              for (int k = 0; k < 2; k++)
                                    {
                                          E_el += lambdaFactorCoul[k] * qq[k] * VV;
                                          DVDL_el += (dLambdaFactor[k] * qq[k]) * VV;
                              }
                        }
#    endif

#    ifdef EL_EWALD_ANY
                        if ((!pairIncluded || r2 < rCutoffCoulSq) && j < nj1)
                        {
                              v_lr = inv_r > 0.0F ? inv_r * erff(r2 * inv_r * beta) : 2.0F * beta * M_FLOAT_1_SQRTPI;
                              if (ai == aj)
                                    v_lr *= 0.5F;
                              for (int k = 0; k < 2; k++)
                              {
                                    E_el -= lambdaFactorCoul[k] * qq[k] * v_lr;
                                    DVDL_el -= (dLambdaFactor[k] * qq[k]) * v_lr;
                              }
                        }
#    endif

                        reduce_fep_energy_warp_shfl(E_lj, E_el, DVDL_lj, DVDL_el,
                              e_lj+lambdaIdx, e_el+lambdaIdx, dvdl_lj+lambdaIdx, dvdl_el+lambdaIdx, tid, c_fullWarpMask);
                  } // end for lambdaIdx in (0, n_lambda)
            //} // end if (j < nj1)
            } // end for (int i = nj0; i < nj1; i += warp_size)
      } // end if (wid_global < nri)
}
#endif /* FUNCTION_DECLARATION_ONLY */

#undef EL_EWALD_ANY
#undef EXCLUSION_FORCES
#undef LJ_EWALD

#undef LJ_COMB
