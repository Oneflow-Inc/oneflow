/*
* Copyright 2009-2016  NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and conditions
* of a form of NVIDIA software license agreement.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

#include "nvToolsExt.h"

#include "cuda.h"
#include "driver_types.h"

#ifndef NVTOOLSEXT_CUDART_V3
#define NVTOOLSEXT_CUDART_V3

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* ========================================================================= */
/** \name Functions for CUDA Resource Naming
*/
/** \addtogroup RESOURCE_NAMING
 * \section RESOURCE_NAMING_CUDART CUDA Runtime Resource Naming
 *
 * This section covers the API functions that allow to annotate CUDA resources
 * with user-provided names.
 *
 * @{
 */

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN 
* \brief Used to build a non-colliding value for resource types separated class
* \version \NVTX_VERSION_2
*/
#define NVTX_RESOURCE_CLASS_CUDART 5
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \brief Resource types for CUDART
*/
typedef enum nvtxResourceCUDARTType_t
{
    NVTX_RESOURCE_TYPE_CUDART_DEVICE = NVTX_RESOURCE_MAKE_TYPE(CUDART, 0), /* int device */
    NVTX_RESOURCE_TYPE_CUDART_STREAM = NVTX_RESOURCE_MAKE_TYPE(CUDART, 1), /* cudaStream_t */
    NVTX_RESOURCE_TYPE_CUDART_EVENT = NVTX_RESOURCE_MAKE_TYPE(CUDART, 2), /* cudaEvent_t */
} nvtxResourceCUDARTType_t;


/* ------------------------------------------------------------------------- */
/** \brief Annotates a CUDA device.
 *
 * Allows the user to associate a CUDA device with a user-provided name.
 *
 * \param device - The id of the CUDA device to name.
 * \param name   - The name of the CUDA device.
 *
 * \version \NVTX_VERSION_1
 * @{ */
NVTX_DECLSPEC void NVTX_API nvtxNameCudaDeviceA(int device, const char* name);
NVTX_DECLSPEC void NVTX_API nvtxNameCudaDeviceW(int device, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates a CUDA stream.
 *
 * Allows the user to associate a CUDA stream with a user-provided name.
 *
 * \param stream - The handle of the CUDA stream to name.
 * \param name   - The name of the CUDA stream.
 *
 * \version \NVTX_VERSION_1
 * @{ */
NVTX_DECLSPEC void NVTX_API nvtxNameCudaStreamA(cudaStream_t stream, const char* name);
NVTX_DECLSPEC void NVTX_API nvtxNameCudaStreamW(cudaStream_t stream, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates a CUDA event.
 *
 * Allows the user to associate a CUDA event with a user-provided name.
 *
 * \param event - The handle of the CUDA event to name.
 * \param name  - The name of the CUDA event.
 *
 * \version \NVTX_VERSION_1
 * @{ */
NVTX_DECLSPEC void NVTX_API nvtxNameCudaEventA(cudaEvent_t event, const char* name);
NVTX_DECLSPEC void NVTX_API nvtxNameCudaEventW(cudaEvent_t event, const wchar_t* name);
/** @} */

/** @} */ /* END RESOURCE_NAMING */

/* ========================================================================= */
#ifdef UNICODE
  #define nvtxNameCudaDevice nvtxNameCudaDeviceW
  #define nvtxNameCudaStream nvtxNameCudaStreamW
  #define nvtxNameCudaEvent  nvtxNameCudaEventW
#else
  #define nvtxNameCudaDevice nvtxNameCudaDeviceA
  #define nvtxNameCudaStream nvtxNameCudaStreamA
  #define nvtxNameCudaEvent  nvtxNameCudaEventA
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifndef NVTX_NO_IMPL
#define NVTX_IMPL_GUARD_CUDART /* Ensure other headers cannot included directly */
#include "nvtxDetail/nvtxImplCudaRt_v3.h"
#undef NVTX_IMPL_GUARD_CUDART
#endif /*NVTX_NO_IMPL*/

#endif /* NVTOOLSEXT_CUDART_V3 */
