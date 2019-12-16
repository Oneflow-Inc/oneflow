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

#ifndef NVTX_IMPL_GUARD_SYNC
#error Never include this file directly -- it is automatically included by nvToolsExtCuda.h (except when NVTX_NO_IMPL is defined).
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef nvtxSyncUser_t (NVTX_API * nvtxDomainSyncUserCreate_impl_fntype)(nvtxDomainHandle_t domain, const nvtxSyncUserAttributes_t* attribs);
typedef void (NVTX_API * nvtxDomainSyncUserDestroy_impl_fntype)(nvtxSyncUser_t handle);
typedef void (NVTX_API * nvtxDomainSyncUserAcquireStart_impl_fntype)(nvtxSyncUser_t handle);
typedef void (NVTX_API * nvtxDomainSyncUserAcquireFailed_impl_fntype)(nvtxSyncUser_t handle);
typedef void (NVTX_API * nvtxDomainSyncUserAcquireSuccess_impl_fntype)(nvtxSyncUser_t handle);
typedef void (NVTX_API * nvtxDomainSyncUserReleasing_impl_fntype)(nvtxSyncUser_t handle);

NVTX_DECLSPEC nvtxSyncUser_t NVTX_API nvtxDomainSyncUserCreate(nvtxDomainHandle_t domain, const nvtxSyncUserAttributes_t* attribs)
{
#ifndef NVTX_DISABLE
    nvtxDomainSyncUserCreate_impl_fntype local = (nvtxDomainSyncUserCreate_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxDomainSyncUserCreate_impl_fnptr;
    if(local!=0)
        return (*local)(domain, attribs);
    else
#endif  /*NVTX_DISABLE*/
        return (nvtxSyncUser_t)0;
}

NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserDestroy(nvtxSyncUser_t handle)
{
#ifndef NVTX_DISABLE
    nvtxDomainSyncUserDestroy_impl_fntype local = (nvtxDomainSyncUserDestroy_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxDomainSyncUserDestroy_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserAcquireStart(nvtxSyncUser_t handle)
{
#ifndef NVTX_DISABLE
    nvtxDomainSyncUserAcquireStart_impl_fntype local = (nvtxDomainSyncUserAcquireStart_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxDomainSyncUserAcquireStart_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserAcquireFailed(nvtxSyncUser_t handle)
{
#ifndef NVTX_DISABLE
    nvtxDomainSyncUserAcquireFailed_impl_fntype local = (nvtxDomainSyncUserAcquireFailed_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxDomainSyncUserAcquireFailed_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserAcquireSuccess(nvtxSyncUser_t handle)
{
#ifndef NVTX_DISABLE
    nvtxDomainSyncUserAcquireSuccess_impl_fntype local = (nvtxDomainSyncUserAcquireSuccess_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxDomainSyncUserAcquireSuccess_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*NVTX_DISABLE*/
}

NVTX_DECLSPEC void NVTX_API nvtxDomainSyncUserReleasing(nvtxSyncUser_t handle)
{
#ifndef NVTX_DISABLE
    nvtxDomainSyncUserReleasing_impl_fntype local = (nvtxDomainSyncUserReleasing_impl_fntype)NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).nvtxDomainSyncUserReleasing_impl_fnptr;
    if(local!=0)
        (*local)(handle);
#endif /*NVTX_DISABLE*/
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
