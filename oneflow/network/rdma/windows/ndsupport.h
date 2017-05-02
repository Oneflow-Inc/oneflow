// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// Net Direct Helper Interface
//

#pragma once

#ifndef _NETDIRECT_H_
#define _NETDIRECT_H_

#include "ndspi.h"


#ifdef __cplusplus
extern "C"
{
#endif  // __cplusplus

#define ND_HELPER_API  __stdcall


//
// Initialization
//
HRESULT ND_HELPER_API
NdStartup(
  VOID
  );

HRESULT ND_HELPER_API
NdCleanup(
  VOID
  );

VOID ND_HELPER_API
NdFlushProviders(
  VOID
  );

//
// Network capabilities
//
#define ND_QUERY_EXCLUDE_EMULATOR_ADDRESSES 0x00000001
#define ND_QUERY_EXCLUDE_NDv1_ADDRESSES   0x00000002
#define ND_QUERY_EXCLUDE_NDv2_ADDRESSES   0x00000004

HRESULT ND_HELPER_API
NdQueryAddressList(
  _In_ DWORD flags,
  _Out_opt_bytecap_post_bytecount_(*pcbAddressList, *pcbAddressList) SOCKET_ADDRESS_LIST* pAddressList,
  _Inout_ SIZE_T* pcbAddressList
  );


HRESULT ND_HELPER_API
NdResolveAddress(
  _In_bytecount_(cbRemoteAddress) const struct sockaddr* pRemoteAddress,
  _In_ SIZE_T cbRemoteAddress,
  _Out_bytecap_(*pcbLocalAddress) struct sockaddr* pLocalAddress,
  _Inout_ SIZE_T* pcbLocalAddress
  );


HRESULT ND_HELPER_API
NdCheckAddress(
  _In_bytecount_(cbAddress) const struct sockaddr* pAddress,
  _In_ SIZE_T cbAddress
  );


HRESULT ND_HELPER_API
NdOpenAdapter(
  _In_ REFIID iid,
  _In_bytecount_(cbAddress) const struct sockaddr* pAddress,
  _In_ SIZE_T cbAddress,
  _Deref_out_ VOID** ppIAdapter
  );


HRESULT ND_HELPER_API
NdOpenV1Adapter(
  _In_bytecount_(cbAddress) const struct sockaddr* pAddress,
  _In_ SIZE_T cbAddress,
  _Deref_out_ INDAdapter** ppIAdapter
  );

#ifdef __cplusplus
}
#endif  // __cplusplus


#endif // _NETDIRECT_H_
