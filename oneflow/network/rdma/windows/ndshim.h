// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// ndshim.h - NetworkDirect v1 to v2 shim, allowing v2 clients
//            use v1 providers.
//


EXTERN_C HRESULT
NdShimOpenAdapter(
    __in REFIID iid,
    __in_bcount(cbAddress) const struct sockaddr* pAddress,
    __in SIZE_T cbAddress,
    __deref_out VOID** ppIAdapter
    )
{
    return ND_NOT_SUPPORTED;
}


template<typename TQp, typename TCq, typename TSge>
class NDIoHelper
{
protected:
    TQp* m_pQp;
    TCq* m_pCq;
    TSge m_rSge;
    TSge* m_pSgl;

protected:
    NDIoHelper() :
        m_pQp( NULL ),
        m_pCq( NULL ),
        m_pSgl( NULL )
    {
        RtlZeroMemory( &m_rSge, sizeof(m_rSge) );
    }

    ~NDIoHelper()
    {
        if( m_pQp != NULL )
        {
            m_pQp->Release();
        }

        if( m_pCq != NULL )
        {
            m_pCq->Release();
        }

        if( m_pSgl != NULL )
        {
            delete[] m_pSgl;
        }
    }

public:
    inline void Receive( __in ND_RESULT* pResult, __in DWORD line )
    {
        HRESULT hr = m_pQp->Receive( pResult, &m_rSge, 1 );
        if( FAILED( hr ) )
        {
            printf( "Receive failed with %08x\n", hr );
            exit( line );
        }
    }

    inline void Send( __in ND_RESULT* pResult, __in DWORD nSge, __in DWORD line )
    {
        HRESULT hr = m_pQp->Send( pResult, m_pSgl, nSge, 0 );
        if( FAILED( hr ) )
        {
            printf( "Send failed with %08x\n", hr );
            exit( line );
        }
    }
};


class NDIoHelperV1 : public NDIoHelper<INDEndpoint, INDCompletionQueue, ND_SGE>
{
    ND_MR_HANDLE m_hMr;

public:
    NDIoHelperV1() :
        m_hMr( NULL )
    {
    }

    void Init(
        __in IND2QueuePair*,
        __in IND2CompletionQueue*,
        __in IND2MemoryRegion*,
        __in void*,
        __in DWORD,
        __in DWORD,
        __in DWORD line
        )
    {
        printf( "NDSPIv1 not supported for self-build.\n" );
        exit( line );
    }

    ~NDIoHelperV1()
    {
    }

    void RefreshQp( __in IND2QueuePair*, __in DWORD line )
    {
        printf( "NDSPIv1 not supported for self-build.\n" );
        exit( line );
    }

    inline void SetSendSge( __in DWORD, __in void*, __in DWORD )
    {
    }

    inline DWORD GetSendSgeLength( __in DWORD )
    {
        return 0;
    }

    DWORD GetResults( __out void**, __out DWORD*, __in DWORD )
    {
        return 0;
    }

public:
    static void OpenAdapter( __in const struct sockaddr_in& addr, __deref_out IND2Adapter** ppAdapter, DWORD line )
    {
        HRESULT hr = NdShimOpenAdapter(
            IID_IND2Adapter,
            reinterpret_cast<const struct sockaddr*>(&addr),
            sizeof(addr),
            reinterpret_cast<VOID**>(ppAdapter)
            );
        if( FAILED( hr ) )
        {
            printf( "NdShimOpenAdapter failed with %08x\n", hr );
            exit( line );
        }
    }
};



class NDIoHelperV2 : public NDIoHelper<IND2QueuePair, IND2CompletionQueue, ND2_SGE>
{
    UINT32 m_Token;

public:
    NDIoHelperV2() :
        m_Token( 0 )
    {
    }

    void Init(
        __in IND2QueuePair* pQp,
        __in IND2CompletionQueue* pCq,
        __in IND2MemoryRegion* pMr,
        __in void* pBuf,
        __in DWORD cbBuf,
        __in DWORD nSge,
        __in DWORD line
        )
    {
        pQp->AddRef();
        m_pQp = pQp;

        pCq->AddRef();
        m_pCq = pCq;

        m_Token = pMr->GetLocalToken();

        //
        // Allocate and setup the SGE for all the transfers.
        // Note that all the transfers overlap, so data verification
        // is not possible.
        //
        m_pSgl = new ND2_SGE[nSge];
        if( m_pSgl == NULL )
        {
            printf( "Failed to allocate SGL.\n" );
            exit( line );
        }

        m_rSge.Buffer = pBuf;
        m_rSge.BufferLength = cbBuf;
        m_rSge.MemoryRegionToken = m_Token;
    }

    ~NDIoHelperV2()
    {
    }

    static void RefreshQp( __in IND2QueuePair*, __in DWORD ){}

    inline void SetSendSge( __in DWORD iSge, __in void* pBuf, __in DWORD cbBuf )
    {
        m_pSgl[iSge].Buffer = pBuf;
        m_pSgl[iSge].BufferLength = cbBuf;
        m_pSgl[iSge].MemoryRegionToken = m_Token;
    }

    inline DWORD GetSendSgeLength( __in DWORD iSge ) const
    {
        return m_pSgl[iSge].BufferLength;
    }

    DWORD GetResults( __out void** pContext, __out DWORD* pcbXfer, __in DWORD line )
    {
        ND2_RESULT result;
        DWORD nResults = m_pCq->GetResults( &result, 1 );

        if( nResults == 0 )
        {
            return 0;
        }

        if( result.Status != ND_SUCCESS )
        {
            if( result.Status != ND_CANCELED )
            {
                printf(
                    "IND2CompletionQueue::GetResults returned result with %08x.\n",
                    result.Status );
                exit( line );
            }

            return 0;
        }

        *pContext = result.RequestContext;
        *pcbXfer = result.BytesTransferred;
        return 1;
    }


    static void OpenAdapter( __in const struct sockaddr_in& addr, __deref_out IND2Adapter** ppAdapter, __in DWORD line )
    {
        HRESULT hr = NdOpenAdapter(
            IID_IND2Adapter,
            reinterpret_cast<const struct sockaddr*>(&addr),
            sizeof(addr),
            reinterpret_cast<VOID**>(ppAdapter)
            );
        if( FAILED( hr ) )
        {
            printf( "NdOpenAdapter failed with %08x\n", hr );
            exit( line );
        }
    }
};

