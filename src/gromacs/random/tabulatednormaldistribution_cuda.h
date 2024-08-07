/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2015,2016,2018,2019,2021, by the GROMACS development team, led by
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

/*! \file
 * \brief Tabulated normal distribution
 *
 * A very fast normal distribution, but with limited resolution.
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \inpublicapi
 * \ingroup module_random
 */

#ifndef GMX_RANDOM_TABULATEDNORMALDISTRIBUTION_H
#define GMX_RANDOM_TABULATEDNORMALDISTRIBUTION_H

#include <cmath>

#include <array>
#include <limits>
#include <memory>

#include "gromacs/math/functions.h"
#include "gromacs/math/utilities.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/real.h"

namespace gmx
{

namespace detail
{

//! Number of bits that determines the resolution of the lookup table for the normal distribution.
constexpr int c_TabulatedNormalDistributionDefaultBits = 14;

} // namespace detail

/*! \brief Tabulated normal random distribution
 *
 *  Random distribution compatible with C++11 distributions - it can be
 *  used with any C++11 random engine.
 *
 *  \tparam tableBits Size of the table, specified in bits. The storage
 *                    space required is sizeof(float)*2^tableBits. To
 *                    keep things sane this is limited to 24 bits.
 *
 *  Some stochastic integrators depend on drawing a lot of normal
 *  distribution random numbers quickly, but in many cases the only
 *  important property is the distribution - given the noise in forces
 *  we do not need very high resolution.
 *  This distribution uses an internal table to return samples from a
 *  normal distribution with limited resolution. By default the table
 *  uses c_TabulatedNormalDistributionDefaultBits bits, but this is
 *  specified with a template parameter.
 *
 *  Since this distribution only uses tableBits bits per value generated,
 *  the values draw from the random engine are used for several results.
 *  To make sure you get a reproducible result when using counter-based
 *  random engines (such as ThreeFry2x64), remember to call the reset()
 *  method to cancel the internal memory of the distribution.
 *
 *  \note For modern NUMA systems, you likely want to use separate
 *        distributions for each thread, and make sure they are initialized
 *        on the CPU where they will run, so the table is placed in that
 *        NUMA memory pool.
 *  \note The finite table resolution means this distribution will NOT
 *        return arbitrarily small/large values, but with e.g. 14 bits
 *        the results are limited to roughly +/- 4 standard deviations.
 */
template<unsigned int tableBits = detail::c_TabulatedNormalDistributionDefaultBits>
class TabulatedNormalDistribution
{
    static_assert(tableBits <= 24,
                  "Normal distribution table is limited to 24bits (64MB in single precision)");

public:
    /*! \brief  Type of normal distribution results */
    typedef float result_type;

    /*! \brief  Normal distribution parameter class (mean and stddev) */
    class param_type
    {
    public:
        /*! \brief The type of distribution the parameters describe */
        typedef TabulatedNormalDistribution distribution_type;

        /*! \brief Constructor. Default is classical distr. with mean 0, stddev 1.
         *
         * \param d_tablePtr Pointer to distribution table on the device.
         * \param mean       Expectation value.
         * \param stddev     Standard deviation.
         *
         */
        __device__ /*explicit*/ param_type(result_type mean = 0.0, result_type stddev = 1.0) :
            mean_(mean), stddev_(stddev)
        {
        }

        /*! \brief Return mean parameter of normal distribution */
        __device__ result_type mean() const { return mean_; }

        /*! \brief Return standard deviation parameter of normal distribution */
        __device__ result_type stddev() const { return stddev_; }

        /*! \brief True if two sets of normal distributions parameters are identical
         *
         * \param x Instance to compare with.
         */
        bool operator==(const param_type& x) const
        {
            return (mean_ == x.mean_ && stddev_ == x.stddev_);
        }

        /*! \brief True if two sets of normal distributions parameters are different.
         *
         * \param x Instance to compare with.
         */
        bool operator!=(const param_type& x) const { return !operator==(x); }

    private:
        /*! \brief Internal storage for mean of normal distribution */
        result_type mean_;
        /*! \brief Internal storage for standard deviation of normal distribution */
        result_type stddev_;
    };

    /*! \brief Construct new normal distribution with specified mean & stdddev.
     *
     *  \param mean    Mean value of tabulated normal distribution
     *  \param stddev  Standard deviation of tabulated normal distribution
     */
    __device__ explicit TabulatedNormalDistribution(const float* __restrict__ d_tablePtr,
                                                    result_type mean   = 0.0,
                                                    result_type stddev = 1.0) :
        table_(d_tablePtr), param_(param_type(mean, stddev)), savedRandomBits_(0), savedRandomBitsLeft_(0)
    {
    }

    /*! \brief Construct new normal distribution from parameter type.
     *
     *  \param param Parameter class containing mean and standard deviation.
     */
    explicit TabulatedNormalDistribution(const float* __restrict__ d_tablePtr, const param_type& param) :
        table_(d_tablePtr), param_(param), savedRandomBits_(0), savedRandomBitsLeft_(0)
    {
    }

    /*! \brief Smallest value that can be generated in normal distrubiton.
     *
     * \note The smallest value is not -infinity with a table, but it
     *       depends on the table resolution. With 14 bits, this is roughly
     *       four standard deviations below the mean.
     */
    result_type min() const { return table_[0]; }

    /*! \brief Largest value that can be generated in normal distribution.
     *
     * \note The largest value is not infinity with a table, but it
     *       depends on the table resolution. With 14 bits, this is roughly
     *       four standard deviations above the mean.
     */
    result_type max() const { return table_[1 << tableBits - 1]; }

    /*! \brief Mean of the present normal distribution */
    result_type mean() const { return param_.mean(); }

    /*! \brief Standard deviation of the present normal distribution */

    result_type stddev() const { return param_.stddev(); }

    /*! \brief The parameter class (mean & stddev) of the normal distribution */
    param_type param() const { return param_; }

    /*! \brief Clear all internal saved random bits from the random engine */
    __device__ void reset() { savedRandomBitsLeft_ = 0; }

    /*! \brief Return normal distribution value specified by internal parameters.
     *
     * \tparam Rng   Random engine type used to provide uniform random bits.
     * \param  g     Random engine of class Rng. For normal GROMACS usage
     *               you likely want to use ThreeFry2x64.
     */
    template<class Rng>
    __device__ result_type operator()(Rng& g)
    {
        return (*this)(g, param_);
    }

    /*! \brief Return normal distribution value specified by given parameters
     *
     * \tparam Rng   Random engine type used to provide uniform random bits.
     * \param  g     Random engine of class Rng. For normal GROMACS usage
     *               you likely want to use ThreeFry2x64.
     * \param  param Parameters used to specify normal distribution.
     */
    template<class Rng>
    __device__ result_type operator()(Rng& g, const param_type& param)
    {
        if (savedRandomBitsLeft_ < tableBits)
        {
            // We do not know whether the generator g returns 64 or 32 bits,
            // since g is not known when we construct this class.
            // To keep things simple, we always draw one random number,
            // store it in our 64-bit value, and set the number of active bits.
            // For tableBits up to 16 this will be as efficient both with 32
            // and 64 bit random engines when drawing multiple numbers
            // (our default value is
            // c_TabulatedNormalDistributionDefaultBits == 14). It
            // also avoids drawing multiple 32-bit random numbers
            // even if we just call this routine for a single
            // result.
            savedRandomBits_     = static_cast<uint64_t>(g());
            savedRandomBitsLeft_ = std::numeric_limits<typename Rng::result_type>::digits;
        }
        result_type value = table_[savedRandomBits_ & ((1ULL << tableBits) - 1)];
        savedRandomBits_ >>= tableBits;
        savedRandomBitsLeft_ -= tableBits;
        return param.mean() + value * param.stddev();
    }

    /*!\brief Check if two tabulated normal distributions have identical states.
     *
     * \param  x     Instance to compare with.
     */
    bool operator==(const TabulatedNormalDistribution<tableBits>& x) const
    {
        return (param_ == x.param_ && savedRandomBits_ == x.savedRandomBits_
                && savedRandomBitsLeft_ == x.savedRandomBitsLeft_);
    }

    /*!\brief Check if two tabulated normal distributions have different states.
     *
     * \param  x     Instance to compare with.
     */
    bool operator!=(const TabulatedNormalDistribution<tableBits>& x) const
    {
        return !operator==(x);
    }

private:
    /*! \brief Parameters of normal distribution (mean and stddev) */
    param_type param_;
    /*! \brief Array with tabulated values of normal distribution */
    const float* table_;
    /*! \brief Saved output from random engine, shifted tableBits right each time */
    uint64_t savedRandomBits_;
    /*! \brief Number of valid bits remaining in savedRandomBits_ */
    unsigned int savedRandomBitsLeft_;

    GMX_DISALLOW_COPY_AND_ASSIGN(TabulatedNormalDistribution);
};

// MSVC does not handle extern template class members correctly even in MSVC 2015,
// so in that case we have to instantiate in every object using it. In addition,
// doxygen is convinced this defines a function (which leads to crashes in our python
// scripts), so to avoid confusion we hide it from doxygen too.
#if !defined(_MSC_VER) && !defined(DOXYGEN)
// Declaration of template specialization
// template<>
// const real TabulatedNormalDistribution<>::table_[1 << detail::c_TabulatedNormalDistributionDefaultBits];

// TODO need a kernel call to do this, or to work out how to fill the
// C array at compile time, or how to revert to using std::array on
// the device, or to implement a simple thing like std::array on the
// device
/* FIXME: Hack to allow calling makeTable() instead of redefining it. */
// int result = TabulatedNormalDistribution<>::makeTable();
#endif

// Instantiation for all tables without specialization
/*
template<class float, unsigned int tableBits>
const float TabulatedNormalDistribution<float, tableBits>::table_[1 << tableBits] =
        TabulatedNormalDistribution<float, tableBits>::makeTable();
*/

} // namespace gmx

#endif // GMX_RANDOM_TABULATEDNORMALDISTRIBUTION_H
