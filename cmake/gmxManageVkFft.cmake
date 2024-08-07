#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2022- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

# Manage VkFFT, GPU FFT library used with OpenCL and SYCL.

function(gmx_manage_vkfft BACKEND_NAME)
    set(GMX_EXTERNAL_VKFFT FALSE CACHE BOOL "Use VkFFT library that is external to GROMACS (ON), or the bundled one (OFF)")
    mark_as_advanced(GMX_EXTERNAL_VKFFT)

    if (NOT GMX_EXTERNAL_VKFFT)
        set(vkfft_DIR ${PROJECT_SOURCE_DIR}/src/external/vkfft)
        set(vkfft_VERSION "internal (1.2.26-b15cb0ca3e884bdb6c901a12d87aa8aadf7637d8) with ${BACKEND_NAME} backend" PARENT_SCOPE)
    else()
        find_path(vkfft_DIR
            NAMES vkFFT.h
            HINTS "${VKFFT_INCLUDE_DIR}"
            DOC "vkFFT directory"
        )
        if(NOT vkfft_DIR)
            message(FATAL_ERROR "External VkFFT requested, but could not be found. Please set VKFFT_INCLUDE_DIR to the directory containing vkFFT.h")
        endif()
        set(vkfft_VERSION "external (from ${vkfft_DIR}) with ${BACKEND_NAME} backend" PARENT_SCOPE)
    endif()

    add_library(VkFFT INTERFACE)
    target_include_directories(VkFFT INTERFACE ${vkfft_DIR})

    # The "-Wcast-qual" warning appears when compiling VkFFT for OpenCL, but not for HIP. It cannot be suppressed.
    gmx_target_interface_warning_suppression(VkFFT "-Wno-unused-parameter" HAS_WARNING_NO_UNUSED_PARAMETER)
    gmx_target_interface_warning_suppression(VkFFT "-Wno-unused-variable" HAS_WARNING_NO_UNUSED_VARIABLE)
    gmx_target_interface_warning_suppression(VkFFT "-Wno-newline-eof" HAS_WARNING_NO_NEWLINE_EOF)
    gmx_target_interface_warning_suppression(VkFFT "-Wno-old-style-cast" HAS_WARNING_NO_OLD_STYLE_CAST)
    gmx_target_interface_warning_suppression(VkFFT "-Wno-zero-as-null-pointer-constant" HAS_WARNING_NO_ZERO_AS_NULL_POINTER_CONSTANT)
    gmx_target_interface_warning_suppression(VkFFT "-Wno-unused-but-set-variable" HAS_WARNING_NO_UNUSED_BUT_SET_VARIABLE)
    gmx_target_interface_warning_suppression(VkFFT "-Wno-sign-compare" HAS_WARNING_NO_SIGN_COMPARE)

    # Backend-specific settings and workarounds
    if (BACKEND_NAME STREQUAL "CUDA")
        target_compile_definitions(VkFFT INTERFACE VKFFT_BACKEND=1)
        # This is not ideal, because it uses some random version of CUDA. See #4621.
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(VkFFT INTERFACE CUDA::cuda_driver CUDA::nvrtc)
        if (NOT GMX_SYCL_HIPSYCL)
            if(NOT DEFINED ENV{GITLAB_CI}) # Don't warn in CI builds
                message(WARNING "The use of VkFFT with CUDA backend is experimental and not intended for production use")
            endif()
            target_link_libraries(VkFFT INTERFACE CUDA::cudart) # Needed only with DPC++
        endif()
    elseif(BACKEND_NAME STREQUAL "HIP")
        target_compile_definitions(VkFFT INTERFACE VKFFT_BACKEND=2)
        # hipFree is marked `nodiscard` but VkFFT ignores it
        gmx_target_interface_warning_suppression(VkFFT "-Wno-unused-result" HAS_WARNING_NO_UNUSED_RESULT)
    elseif(BACKEND_NAME STREQUAL "OpenCL")
        target_compile_definitions(VkFFT INTERFACE VKFFT_BACKEND=3)
    elseif(BACKEND_NAME STREQUAL "LevelZero")
        target_compile_definitions(VkFFT INTERFACE VKFFT_BACKEND=4)
    else()
        message(FATAL_ERROR "Unknown VkFFT backend name ${BACKEND_NAME}")
    endif()

endfunction()

