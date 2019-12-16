/* This file was procedurally generated!  Do not modify this file by hand.  */

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

#ifndef NVTX_IMPL_GUARD_CUDART
#error Never include this file directly -- it is automatically included by nvToolsExtCudaRt.h (except when NVTX_NO_IMPL is defined).
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void (NVTX_API * nvtxNameCudaDeviceA_impl_fntype)(int device, const char* name);
typedef void (NVTX_API * nvtxNameCudaDeviceW_impl_fntype)(int device, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCudaStreamA_impl_fntype)(cudaStream_t stream, const char* name);
typedef void (NVTX_API * nvtxNameCudaStreamW_impl_fntype)(cudaStream_t stream, const wchar_t* name);
typedef void (NVTX_API * nvtxNameCudaEventA_impl_fntype)(cudaEvent_t event, const char* name);
typedef void (NVTX_API * nvtxNameCudaEventW_impl_fntype)(cudaEvent_t event, const wchar_t* name);

NVTX_DECLSPEC void NVTX_API nvtxNameCudaDeviceA(int device, const char* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCudaDeviceA_impl_fntype local = (nvtxNameCudaDeviceA_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCudaDeviceA_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCudaDeviceW(int device, const wchar_t* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCudaDeviceW_impl_fntype local = (nvtxNameCudaDeviceW_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCudaDeviceW_impl_fnptr;
    if(local!=0)
        (*local)(device, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCudaStreamA(cudaStream_t stream, const char* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCudaStreamA_impl_fntype local = (nvtxNameCudaStreamA_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCudaStreamA_impl_fnptr;
    if(local!=0)
        (*local)(stream, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCudaStreamW(cudaStream_t stream, const wchar_t* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCudaStreamW_impl_fntype local = (nvtxNameCudaStreamW_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCudaStreamW_impl_fnptr;
    if(local!=0)
        (*local)(stream, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCudaEventA(cudaEvent_t event, const char* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCudaEventA_impl_fntype local = (nvtxNameCudaEventA_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCudaEventA_impl_fnptr;
    if(local!=0)
        (*local)(event, name);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxNameCudaEventW(cudaEvent_t event, const wchar_t* name)
{
#ifndef NVTX_DISABLE
    nvtxNameCudaEventW_impl_fntype local = (nvtxNameCudaEventW_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxNameCudaEventW_impl_fnptr;
    if(local!=0)
        (*local)(event, name);
#endif /*NVTX_DISABLE*/
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

