// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//

#include <winsock2.h>
#include <ws2tcpip.h>
#include <ws2spi.h>
#include "initguid.h"
#include "ndsupport.h"


typedef HRESULT
(*DLLGETCLASSOBJECT)(
    REFCLSID rclsid,
    REFIID rrid,
    LPVOID* ppv
    );

typedef HRESULT
(*DLLCANUNLOADNOW)(void);


HMODULE g_hProvider = NULL;
INDProvider* g_pIProvider = NULL;
DLLCANUNLOADNOW g_pfnDllCanUnloadNow = NULL;


static WCHAR* GetProviderPath( WSAPROTOCOL_INFOW* pProtocol )
{
    INT pathLen;
    INT ret, err;
    WCHAR* pPath;
    WCHAR* pPathEx;

    // Get the path length for the provider DLL.
    pathLen = 0;
    ret = ::WSCGetProviderPath( &pProtocol->ProviderId, NULL, &pathLen, &err );

    if( err != WSAEFAULT || pathLen == 0 )
    {
        return NULL;
    }

    pPath = static_cast<WCHAR*>(
        ::HeapAlloc( ::GetProcessHeap(), 0, sizeof(WCHAR) * pathLen )
        );
    if( pPath == NULL )
    {
        return NULL;
    }

    ret = ::WSCGetProviderPath( &pProtocol->ProviderId, pPath, &pathLen, &err );
    if( ret != 0 )
    {
        ::HeapFree( ::GetProcessHeap(), 0, pPath );
        return NULL;
    }

    pathLen = ::ExpandEnvironmentStringsW( pPath, NULL, 0 );
    if( pathLen == 0 )
    {
        ::HeapFree( ::GetProcessHeap(), 0, pPath );
        return NULL;
    }

    pPathEx = static_cast<WCHAR*>(
        ::HeapAlloc( ::GetProcessHeap(), 0, sizeof(WCHAR) * pathLen )
        );
    if( pPathEx == NULL )
    {
        ::HeapFree( ::GetProcessHeap(), 0, pPath );
        return NULL;
    }

    ret = ::ExpandEnvironmentStringsW( pPath, pPathEx, pathLen );

    // We don't need the un-expanded path anymore.
    ::HeapFree( ::GetProcessHeap(), 0, pPath );

    if( ret != pathLen )
    {
        ::HeapFree( ::GetProcessHeap(), 0, pPathEx );
        return NULL;
    }

    return pPathEx;
}


static HRESULT LoadProvider( WSAPROTOCOL_INFOW* pProtocol )
{
    WCHAR* pPath = ::GetProviderPath( pProtocol );
    if( pPath == NULL )
    {
        return ND_UNSUCCESSFUL;
    }

    g_hProvider = ::LoadLibraryW( pPath );

    ::HeapFree( ::GetProcessHeap(), 0, pPath );

    if( g_hProvider == NULL )
    {
        return HRESULT_FROM_WIN32( ::GetLastError() );
    }

    DLLGETCLASSOBJECT pfnDllGetClassObject = reinterpret_cast<DLLGETCLASSOBJECT>(
        ::GetProcAddress( g_hProvider, "DllGetClassObject" )
        );
    if( pfnDllGetClassObject == NULL )
    {
        return HRESULT_FROM_WIN32( ::GetLastError() );
    }

    g_pfnDllCanUnloadNow = reinterpret_cast<DLLCANUNLOADNOW>(
        ::GetProcAddress( g_hProvider, "DllCanUnloadNow" )
        );
    if( g_pfnDllCanUnloadNow == NULL )
    {
        return HRESULT_FROM_WIN32( ::GetLastError() );
    }

    IClassFactory* pClassFactory;
    HRESULT hr = pfnDllGetClassObject(
        pProtocol->ProviderId,
        IID_IClassFactory,
        reinterpret_cast<void**>(&pClassFactory)
        );
    if( FAILED(hr) )
    {
        return hr;
    }

    hr = pClassFactory->CreateInstance(
        NULL,
        IID_INDProvider,
        reinterpret_cast<void**>(&g_pIProvider)
        );

    // Now that we asked for the provider, we don't need the class factory.
    pClassFactory->Release();
    return hr;
}


static HRESULT Init()
{
    // Enumerate the provider catalog, find the first ND provider and load it.
    DWORD len = 0;
    INT err;
    INT ret = ::WSCEnumProtocols( NULL, NULL, &len, &err );
    if( ret != SOCKET_ERROR || err != WSAENOBUFS )
    {
        return ND_INTERNAL_ERROR;
    }

    WSAPROTOCOL_INFOW* pProtocols = static_cast<WSAPROTOCOL_INFOW*>(
        ::HeapAlloc( ::GetProcessHeap(), 0, len )
        );
    if( pProtocols == NULL )
    {
        return ND_NO_MEMORY;
    }

    ret = ::WSCEnumProtocols( NULL, pProtocols, &len, &err );
    if( ret == SOCKET_ERROR )
    {
        ::HeapFree( ::GetProcessHeap(), 0, pProtocols );
        return ND_INTERNAL_ERROR;
    }

    HRESULT hr;
    for( DWORD i = 0; i < len / sizeof(WSAPROTOCOL_INFOW); i++ )
    {
#define ServiceFlags1Flags (XP1_GUARANTEED_DELIVERY | XP1_GUARANTEED_ORDER | \
    XP1_MESSAGE_ORIENTED | XP1_CONNECT_DATA)

        if( (pProtocols[i].dwServiceFlags1 & ServiceFlags1Flags) !=
            ServiceFlags1Flags )
        {
            continue;
        }

        if( pProtocols[i].iAddressFamily != AF_INET &&
            pProtocols[i].iAddressFamily != AF_INET6 )
        {
            continue;
        }

        if( pProtocols[i].iSocketType != -1 )
        {
            continue;
        }

        if( pProtocols[i].iProtocol != 0 )
        {
            continue;
        }

        if( pProtocols[i].iProtocolMaxOffset != 0 )
        {
            continue;
        }

        hr = ::LoadProvider( &pProtocols[i] );
    }
    ::HeapFree( ::GetProcessHeap(), 0, pProtocols );

    return hr;
}


EXTERN_C HRESULT ND_HELPER_API
NdStartup(
    VOID
    )
{
    int ret;
    WSADATA data;

    ret = ::WSAStartup( MAKEWORD(2, 2), &data );
    if( ret != 0 )
    {
        return HRESULT_FROM_WIN32( ret );
    }

    HRESULT hr = Init();
    if( FAILED( hr ) )
    {
        NdCleanup();
    }

    return hr;
}


EXTERN_C HRESULT ND_HELPER_API
NdCleanup(
    VOID
    )
{
    if( g_pIProvider != NULL )
    {
        g_pIProvider->Release();
        g_pIProvider = NULL;
    }

    if( g_hProvider != NULL )
    {
        ::FreeLibrary( g_hProvider );
        g_hProvider = NULL;
    }

    ::WSACleanup();

    return S_OK;
}


EXTERN_C VOID ND_HELPER_API
NdFlushProviders(
    VOID
    )
{
    return;
}


EXTERN_C HRESULT ND_HELPER_API
NdQueryAddressList(
    _In_ DWORD Flags,
    _Out_opt_bytecap_post_bytecount_(*pcbAddressList, *pcbAddressList) SOCKET_ADDRESS_LIST* pAddressList,
    _Inout_ SIZE_T* pcbAddressList
    )
{
    UNREFERENCED_PARAMETER( Flags );

    if( g_pIProvider == NULL )
    {
        return ND_DEVICE_NOT_READY;
    }

    return g_pIProvider->QueryAddressList( pAddressList, pcbAddressList );
}


EXTERN_C HRESULT ND_HELPER_API
NdResolveAddress(
    _In_bytecount_(cbRemoteAddress) const struct sockaddr* pRemoteAddress,
    _In_ SIZE_T cbRemoteAddress,
    _Out_bytecap_(*pcbLocalAddress) struct sockaddr* pLocalAddress,
    _Inout_ SIZE_T* pcbLocalAddress
    )
{
    SIZE_T len;

    //
    // Cap to max DWORD value.  This has the added benefit of zeroing the upper
    // bits on 64-bit platforms, so that the returned value is correct.
    //
    if( *pcbLocalAddress > UINT_MAX )
    {
        *pcbLocalAddress = UINT_MAX;
    }

    // We store the original length so we can distinguish from different
    // errors that return WSAEFAULT.
    len = *pcbLocalAddress;

    // Create a socket for address changes.
    SOCKET s = ::WSASocket( AF_INET, SOCK_STREAM, 0, NULL, 0, WSA_FLAG_OVERLAPPED );
    if( s == INVALID_SOCKET )
    {
        return ND_INSUFFICIENT_RESOURCES;
    }

    int ret = ::WSAIoctl(
        s,
        SIO_ROUTING_INTERFACE_QUERY,
        const_cast<sockaddr*>(pRemoteAddress),
        static_cast<DWORD>(cbRemoteAddress),
        pLocalAddress,
        static_cast<DWORD>(len),
        reinterpret_cast<DWORD*>(pcbLocalAddress),
        NULL,
        NULL
        );

    if( ret == SOCKET_ERROR )
    {
        switch( ::GetLastError() )
        {
        case WSAEFAULT:
            if( len < *pcbLocalAddress )
            {
                return ND_BUFFER_OVERFLOW;
            }

            __fallthrough;
        default:
            return ND_UNSUCCESSFUL;
        case WSAEINVAL:
            return ND_INVALID_ADDRESS;
        case WSAENETUNREACH:
        case WSAENETDOWN:
            return ND_NETWORK_UNREACHABLE;
        }
    }

    return ND_SUCCESS;
}


EXTERN_C HRESULT ND_HELPER_API
NdCheckAddress(
    _In_bytecount_(cbAddress) const struct sockaddr* pAddress,
    _In_ SIZE_T cbAddress
    )
{
    INDAdapter* pIAdapter;

    HRESULT hr = NdOpenV1Adapter( pAddress, cbAddress, &pIAdapter );
    if( SUCCEEDED( hr ) )
    {
        pIAdapter->Release();
    }
    return hr;
}


EXTERN_C HRESULT ND_HELPER_API
NdOpenAdapter(
    _In_ REFIID iid,
    _In_bytecount_(cbAddress) const struct sockaddr* pAddress,
    _In_ SIZE_T cbAddress,
    _Deref_out_ VOID** ppIAdapter
    )
{
    return ND_NOT_SUPPORTED;
}


EXTERN_C HRESULT ND_HELPER_API
NdOpenV1Adapter(
    _In_bytecount_(cbAddress) const struct sockaddr* pAddress,
    _In_ SIZE_T cbAddress,
    _Deref_out_ INDAdapter** ppIAdapter
    )
{
    if( g_pIProvider == NULL )
    {
        return ND_DEVICE_NOT_READY;
    }

    return g_pIProvider->OpenAdapter( pAddress, cbAddress, ppIAdapter );
}

