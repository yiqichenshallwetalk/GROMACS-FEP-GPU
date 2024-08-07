/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
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
 *
 * \brief Implements Langevin (SD) integrator using CUDA
 *
 * This file contains implementation of the Langevin (SD) integrator
 * using CUDA, including class initialization, data-structures management
 * and GPU kernel.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include <assert.h>
#include <stdio.h>

#include <cmath>

#include <algorithm>

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/random/tabulatednormaldistribution_cuda.h"
#include "gromacs/random/threefry_cuda.h"
#include "gromacs/utility/arrayref.h"

#include "langevin_gpu.h"

namespace gmx
{

/*!\brief Number of CUDA threads in a block
 *
 * \todo Check if using smaller block size will lead to better performance.
 */
constexpr static int c_threadsPerBlock = 256;
//! Maximum number of threads in a block (for __launch_bounds__)
constexpr static int c_maxThreadsPerBlock = c_threadsPerBlock;
/*! \brief Main kernel for SD integrator.
 *
 *  The coordinates and velocities are updated on the GPU. Also saves the intermediate values of the coordinates for
 *  further use in constraints.
 *
 *  Each GPU thread works with a single particle. Empty declaration is needed to
 *  avoid "no previous prototype for function" clang warning.
 *
 *  \todo Check if the force should be set to zero here.
 *  \todo This kernel can also accumulate incidental temperatures for each atom.
 *
 * \tparam        updateType         Langevin integrator update type. The integration is divided in
 *                                   three steps if there are constraints:
 *                                   SDUpdate::ForcesOnly, Constraints, SDUpdate::FrictionAndNoiseOnly
 *                                   If there are no constraints it is only SDUpdate::Combined. Currently,
 *                                   SDUpdate::Combined is not supported on GPU.
 * \param[in]     numAtoms           Total number of atoms.
 * \param[in,out] gm_x               Coordinates to update upon integration.
 * \param[out]    gm_xp              A copy of the coordinates before the integration (for constraints).
 * \param[in,out] gm_v               Velocities to update.
 * \param[in]     gm_f               Atomic forces.
 * \param[in]     gm_inverseMasses   Reciprocal masses.
 * \param[in]     dt                 Timestep.
 * \param[in]     gm_tempCouplGroups Mapping of atoms into temperate coupling groups.
 */
template<SDUpdate updateType>
__launch_bounds__(c_maxThreadsPerBlock) __global__
        void langevin_kernel(const int numAtoms,
                             float3* __restrict__ gm_x,
                             float3* __restrict__ gm_xp,
                             float3* __restrict__ gm_v,
                             const float3* __restrict__ gm_f,
                             const float* __restrict__ gm_inverseMasses,
                             const float dt,
                             const int   seed,
                             const int   step,
                             const unsigned short* __restrict__ gm_tempCouplGroups,
                             const float* __restrict__ gm_sdSigmaV,
                             const float* __restrict__ gm_sdConstEm,
                             const float* __restrict__ gm_distributionTable)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex < numAtoms)
    {
        // Even 0 bits internal counter gives 2x64 ints (more than enough for three table lookups)
        gmx::ThreeFry2x64<0> rng(seed, gmx::RandomDomain::UpdateCoordinates);
        gmx::TabulatedNormalDistribution<c_normalDistributionTableBits> dist(gm_distributionTable);
        rng.restart(step, threadIndex);
        dist.reset();

        float3 distByDim;
        if (updateType == SDUpdate::FrictionAndNoiseOnly)
        {
            distByDim.x = dist(rng);
            distByDim.y = dist(rng);
            distByDim.z = dist(rng);
        }

        float3 x                     = gm_x[threadIndex];
        float3 v                     = gm_v[threadIndex];
        float3 f                     = gm_f[threadIndex];
        float  inverseMass           = gm_inverseMasses[threadIndex];
        float  inverseSqrtMass       = sqrt(inverseMass);
        float  inverseMassDt         = inverseMass * dt;
        int    temperatureCouplGroup = gm_tempCouplGroups[threadIndex];
        float  sdSigmaV              = gm_sdSigmaV[temperatureCouplGroup];
        float  sdConstEm             = gm_sdConstEm[temperatureCouplGroup];


        if (updateType != SDUpdate::FrictionAndNoiseOnly)
        {
            // Swapping places for xp and x so that the x will contain the updated coordinates and xp - the
            // coordinates before update. This should be taken into account when (if) constraints are applied
            // after the update: x and xp have to be passed to constraints in the 'wrong' order.
            // TODO: Issue #3727
            gm_xp[threadIndex] = x;
        }

        if (updateType == SDUpdate::ForcesOnly)
        {
            float3 vn = v + f * inverseMassDt;
            v         = vn;
            x += v * dt;
        }
        else if (updateType == SDUpdate::FrictionAndNoiseOnly)
        {
            float3 vn = v;
            v         = vn * sdConstEm + inverseSqrtMass * sdSigmaV * distByDim;
            // The previous phase already updated the
            // positions with a full v*dt term that must
            // now be half removed.
            x += 0.5 * (v - vn) * dt;
        }
        else
        {
            float3 vn = v + f * inverseMassDt;
            v += f * inverseMassDt;
            v = v * sdConstEm + inverseSqrtMass * sdSigmaV * distByDim;
            x += 0.5 * (v + vn) * dt;
        }
        gm_v[threadIndex] = v;
        gm_x[threadIndex] = x;
    }
    return;
}

/*! \brief Select templated kernel.
 *
 * Returns pointer to a CUDA kernel based on the type of SD integration.
 *
 * \param[in]  updateType   Langevin integrator update type. The integration is divided in
 *                          three steps if there are constraints:
 *                          SDUpdate::ForcesOnly, Constraints, SDUpdate::FrictionAndNoiseOnly
 *                          If there are no constraints it is only SDUpdate::Combined.
 *
 * \return                  Pointer to CUDA kernel
 */
inline auto selectLangevinKernelPtr(const SDUpdate updateType)
{
    GMX_ASSERT(updateType == SDUpdate::ForcesOnly || updateType == SDUpdate::FrictionAndNoiseOnly,
               "Langevin integrator on GPU cannot do the update in one step, even if there are no "
               "constraints.");

    auto kernelPtr = langevin_kernel<SDUpdate::ForcesOnly>;

    if (updateType == SDUpdate::FrictionAndNoiseOnly)
    {
        kernelPtr = langevin_kernel<SDUpdate::FrictionAndNoiseOnly>;
    }
    return kernelPtr;
}

void LangevinGpu::integrate(DeviceBuffer<Float3>       d_x,
                            DeviceBuffer<Float3>       d_xp,
                            DeviceBuffer<Float3>       d_v,
                            const DeviceBuffer<Float3> d_f,
                            const real                 dt,
                            const int                  seed,
                            const int                  step,
                            const SDUpdate             updateType)
{

    ensureNoPendingDeviceError("In CUDA version of Langevin integrator");
    GMX_ASSERT(updateType == SDUpdate::ForcesOnly || updateType == SDUpdate::FrictionAndNoiseOnly,
               "Langevin integrator on GPU cannot do the update in one step, even if there are no "
               "constraints.");

    auto kernelPtr = selectLangevinKernelPtr(updateType);

    // Checking the buffer types against the kernel argument types
    static_assert(sizeof(*d_inverseMasses_) == sizeof(float), "Incompatible types");
    const auto kernelArgs = prepareGpuKernelArguments(kernelPtr,
                                                      kernelLaunchConfig_,
                                                      &numAtoms_,
                                                      asFloat3Pointer(&d_x),
                                                      asFloat3Pointer(&d_xp),
                                                      asFloat3Pointer(&d_v),
                                                      asFloat3Pointer(&d_f),
                                                      &d_inverseMasses_,
                                                      &dt,
                                                      &seed,
                                                      &step,
                                                      &d_tempCouplGroups_,
                                                      &d_sdSigmaV_,
                                                      &d_sdConstEm_,
                                                      &d_distributionTable_);
    launchGpuKernel(kernelPtr, kernelLaunchConfig_, deviceStream_, nullptr, "langevin_kernel", kernelArgs);

    return;
}

LangevinGpu::LangevinGpu(const DeviceContext& deviceContext,
                         const DeviceStream&  deviceStream,
                         const int            numTempCouplGroups,
                         const float          delta_t,
                         const float*         ref_t,
                         const float*         tau_t) :
    deviceContext_(deviceContext), deviceStream_(deviceStream), numTempCouplGroups_(numTempCouplGroups)
{
    numAtoms_ = 0;

    kernelLaunchConfig_.blockSize[0]     = c_threadsPerBlock;
    kernelLaunchConfig_.blockSize[1]     = 1;
    kernelLaunchConfig_.blockSize[2]     = 1;
    kernelLaunchConfig_.sharedMemorySize = 0;

    std::vector<float> sdConstEm(numTempCouplGroups);
    std::vector<float> sdSigmaV(numTempCouplGroups);

    for (int i = 0; i < numTempCouplGroups; i++)
    {
        /* Compared to the CPU version we lose precision here by using float instead of double. */
        if (tau_t[i] > 0)
        {
            sdConstEm[i] = exp(-delta_t / tau_t[i]);
        }
        else
        {
            /* No friction and noise on this group */
            sdConstEm[i] = 1;
        }

        real kT = gmx::c_boltz * ref_t[i];
        /* The mass is accounted for later, since this differs per atom */
        sdSigmaV[i] = sqrt(kT * (1 - sdConstEm[i] * sdConstEm[i]));
    }
    reallocateDeviceBuffer(
            &d_sdSigmaV_, numTempCouplGroups, &numSdSigmaV_, &numSdSigmaVAlloc_, deviceContext_);
    copyToDeviceBuffer(
            &d_sdSigmaV_, sdSigmaV.data(), 0, numTempCouplGroups, deviceStream_, GpuApiCallBehavior::Sync, nullptr);
    reallocateDeviceBuffer(
            &d_sdConstEm_, numTempCouplGroups, &numSdConstEm_, &numSdConstEmAlloc_, deviceContext_);
    copyToDeviceBuffer(
            &d_sdConstEm_, sdConstEm.data(), 0, numTempCouplGroups, deviceStream_, GpuApiCallBehavior::Sync, nullptr);

    makeDistributionTable();
}

LangevinGpu::~LangevinGpu()
{
    freeDeviceBuffer(&d_inverseMasses_);
    freeDeviceBuffer(&d_tempCouplGroups_);
    freeDeviceBuffer(&d_sdSigmaV_);
    freeDeviceBuffer(&d_sdConstEm_);
}

void LangevinGpu::set(const int                            numAtoms,
                      const ArrayRef<const real>           inverseMasses,
                      const ArrayRef<const unsigned short> tempCouplGroups)
{
    numAtoms_                       = numAtoms;
    kernelLaunchConfig_.gridSize[0] = (numAtoms_ + c_threadsPerBlock - 1) / c_threadsPerBlock;

    reallocateDeviceBuffer(
            &d_inverseMasses_, numAtoms_, &numInverseMasses_, &numInverseMassesAlloc_, deviceContext_);
    copyToDeviceBuffer(
            &d_inverseMasses_, inverseMasses.data(), 0, numAtoms_, deviceStream_, GpuApiCallBehavior::Sync, nullptr);

    reallocateDeviceBuffer(
            &d_tempCouplGroups_, numAtoms_, &numTempCouplGroups_, &numTempCouplGroupsAlloc_, deviceContext_);
    if (!tempCouplGroups.empty())
    {
        copyToDeviceBuffer(&d_tempCouplGroups_,
                           tempCouplGroups.data(),
                           0,
                           numAtoms_,
                           deviceStream_,
                           GpuApiCallBehavior::Sync,
                           nullptr);
    }
    else
    {
        std::vector<unsigned short> dummyTempCouplGroups(numAtoms_, 0);
        copyToDeviceBuffer(&d_tempCouplGroups_,
                           dummyTempCouplGroups.data(),
                           0,
                           numAtoms_,
                           deviceStream_,
                           GpuApiCallBehavior::Sync,
                           nullptr);
    }

    reallocateDeviceBuffer(&d_distributionTable_,
                           1 << c_normalDistributionTableBits,
                           &sizeOfDistributionTable_,
                           &sizeOfDistributionTableAlloc_,
                           deviceContext_);
    copyToDeviceBuffer(&d_distributionTable_,
                       distributionTable_,
                       0,
                       1 << c_normalDistributionTableBits,
                       deviceStream_,
                       GpuApiCallBehavior::Sync,
                       nullptr);
}

void LangevinGpu::makeDistributionTable()
{
    /* Fill the table with the integral of a gaussian distribution, which
     * corresponds to the inverse error function.
     * We avoid integrating a gaussian numerically, since that leads to
     * some loss-of-precision which also accumulates so it is worse for
     * larger indices in the table. */
    constexpr std::size_t tableSize   = 1 << c_normalDistributionTableBits;
    constexpr std::size_t halfSize    = tableSize / 2;
    constexpr double      invHalfSize = 1.0 / halfSize;

    // Fill in all but the extremal entries of the table
    for (std::size_t i = 0; i < halfSize - 1; i++)
    {
        double r = (i + 0.5) * invHalfSize;
        double x = std::sqrt(2.0) * erfinv(r);

        distributionTable_[halfSize - 1 - i] = -x;
        distributionTable_[halfSize + i]     = x;
    }
    // We want to fill in the extremal table entries with
    // values that make the total variance equal to 1, so
    // measure the variance by summing the squares of the
    // other values of the distribution, starting from the
    // smallest values.
    double sumOfSquares = 0;
    for (std::size_t i = 1; i < halfSize; i++)
    {
        double value = distributionTable_[i];
        sumOfSquares += value * value;
    }
    double missingVariance = 1.0 - 2.0 * sumOfSquares / tableSize;
    GMX_RELEASE_ASSERT(missingVariance > 0,
                       "Incorrect computation of tabulated normal distribution");
    double extremalValue              = std::sqrt(0.5 * missingVariance * tableSize);
    distributionTable_[0]             = -extremalValue;
    distributionTable_[tableSize - 1] = extremalValue;
}

} // namespace gmx
