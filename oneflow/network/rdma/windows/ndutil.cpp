// This is the main DLL file.


#include "ndutil.h"

//initializer
Nd2TestBase::Nd2TestBase() :
m_pAdapter( NULL ),
	m_pMr( NULL ),
	m_pCq( NULL ),
	m_pQp( NULL ),
	m_pConnector( NULL ),
	m_hAdapterFile( NULL ),
	m_Buf( NULL ),
	m_pMw(NULL)
{
	RtlZeroMemory( &m_Ov, sizeof(m_Ov ) );
}

//tear down
Nd2TestBase::~Nd2TestBase()
{
	if( m_pMr != NULL )
	{
		m_pMr->Release();
	}

	if( m_pCq != NULL )
	{
		m_pCq->Release();
	}

	if( m_pQp != NULL )
	{
		m_pQp->Release();
	}

	if( m_pConnector != NULL )
	{
		m_pConnector->Release();
	}

	if( m_hAdapterFile != NULL )
	{
		CloseHandle( m_hAdapterFile );
	}

	if( m_pAdapter != NULL )
	{
		m_pAdapter->Release();
	}

	if( m_Ov.hEvent != NULL )
	{
		CloseHandle( m_Ov.hEvent );
	}

	if( m_Buf != NULL )
	{
		delete[] m_Buf;
	}
}

void Nd2TestBase::CreateMR(HRESULT expectedResult, const char* errorMessage)
{

	HRESULT hr = m_pAdapter->CreateMemoryRegion(
		IID_IND2MemoryRegion,
		m_hAdapterFile,
		reinterpret_cast<VOID**>( &m_pMr )
		);

	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);

}



void Nd2TestBase::RegisterDataBuffer(DWORD bufferLength,ULONG type,HRESULT expectedResult, const char* errorMessage)
{
	m_Buf_Len = bufferLength;
	m_Buf = new (std::nothrow) char[m_Buf_Len];
	if( m_Buf == NULL )
	{
		printf( "Failed to allocate buffer.\n" );
		exit( __LINE__ );
	}

	HRESULT hr = m_pMr->Register(
		m_Buf,
		m_Buf_Len,
		type,
		&m_Ov
		);
	if( hr == ND_PENDING )
	{
		hr = m_pMr->GetOverlappedResult( &m_Ov, TRUE );
	}
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}


void Nd2TestBase::CreateMW(HRESULT expectedResult, const char* errorMessage)
{
	HRESULT hr = m_pAdapter->CreateMemoryWindow(
		IID_IND2MemoryWindow,
		reinterpret_cast<VOID**>( &m_pMw )
		);

	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

void Nd2TestBase::InvalidateMW(HRESULT expectedResult, const char* errorMessage)
{
	HRESULT hr;
	hr = m_pQp->Invalidate(NULL,m_pMw,0);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

void Nd2TestBase::Bind(DWORD bufferLength,ULONG type,HRESULT expectedResult, const char* errorMessage)
{

	ND2_RESULT pResult[2];
	HRESULT hr = m_pQp->Bind(
		pResult,
		m_pMr,
		m_pMw,
		m_Buf,
		bufferLength,
		type
		);
	Nd2TestBase::WaitForCompletion();
}

void Nd2TestBase::CreateCQ(DWORD depth,HRESULT expectedResult, const char* errorMessage)
{

	HRESULT hr = m_pAdapter->CreateCompletionQueue(
		IID_IND2CompletionQueue,
		m_hAdapterFile,
		depth,
		0,
		0,
		reinterpret_cast<VOID**>( &m_pCq )
		);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);

}

void Nd2TestBase::CreateConnector(HRESULT expectedResult, const char* errorMessage)
{
	HRESULT hr = m_pAdapter->CreateConnector(
		IID_IND2Connector,
		m_hAdapterFile,
		reinterpret_cast<VOID**>( &m_pConnector )
		);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);

}

void Nd2TestBase::CreateQueuePair(DWORD queueDepth,DWORD nSge,HRESULT expectedResult, const char* errorMessage)
{

	HRESULT hr = m_pAdapter->CreateQueuePair(
		IID_IND2QueuePair,
		m_pCq,
		m_pCq,
		NULL,
		queueDepth,
		1,
		nSge,
		nSge,
		0,
		reinterpret_cast<VOID**>( &m_pQp )
		);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}


void Nd2TestBase::CreateQueuePair(DWORD receiveQueueDepth,DWORD initiatorQueueDepth,DWORD maxReceiveRequestSge, DWORD maxInitiatorRequestSge,HRESULT expectedResult, const char* errorMessage)
{

	HRESULT hr = m_pAdapter->CreateQueuePair(
		IID_IND2QueuePair,
		m_pCq,
		m_pCq,
		NULL,
		maxReceiveRequestSge,
		initiatorQueueDepth,
		maxReceiveRequestSge,
		maxInitiatorRequestSge,
		0,
		reinterpret_cast<VOID**>( &m_pQp )
		);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

void Nd2TestBase::Init(_In_ const struct sockaddr_in& v4Src)
{
	HRESULT hr = NdOpenAdapter(
		IID_IND2Adapter,
		reinterpret_cast<const struct sockaddr*>(&v4Src),
		sizeof(v4Src),
		reinterpret_cast<void**>( &m_pAdapter )
		);
	if( FAILED( hr ) )
	{
		hr = NdShimOpenAdapter(
			IID_IND2Adapter,
			reinterpret_cast<const struct sockaddr*>(&v4Src),
			sizeof(v4Src),
			reinterpret_cast<void**>( &m_pAdapter )
			);
		if( FAILED( hr ) )
		{
          LogErrorExit("Failed open adapter.\n",__LINE__);
		}
	}

	m_Ov.hEvent = CreateEvent( NULL, FALSE, FALSE, NULL );
	if( m_Ov.hEvent == NULL )
	{
		LogErrorExit("Failed to allocate event for overlapped operations.\n",__LINE__);
	}

	//
	// Get the file handle for overlapped operations on this adapter.
	//
	hr = m_pAdapter->CreateOverlappedFile( &m_hAdapterFile );
	if( FAILED( hr ) )
	{
		LogErrorExit(hr,"IND2Adapter::CreateOverlappedFile failed",__LINE__);
	}

}


void Nd2TestBase::DisconnectConnector()
{
	HRESULT hr = m_pConnector->Disconnect( &m_Ov );
}


void Nd2TestBase::DeregisterMemory()
{
	HRESULT hr = m_pMr->Deregister( &m_Ov );
}

void Nd2TestBase::GetResult(HRESULT expectedResult,const char* errorMessage )
{
	HRESULT hr = m_pCq->GetOverlappedResult( &m_Ov, TRUE );
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

void Nd2TestBase::Shutdown()
{
	Nd2TestBase::DisconnectConnector();
	Nd2TestBase::DeregisterMemory();
}

void Nd2TestBase::PostReceive(const ND2_SGE* Sge,const DWORD nSge,HRESULT expectedResult,const char* errorMessage)
{
	HRESULT hr = m_pQp->Receive( &m_Results[RECV], Sge, nSge );
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

void Nd2TestBase::Write(const ND2_SGE* Sge,const ULONG nSge,UINT64 remoteAddress, UINT32 remoteToken,DWORD flag,HRESULT expectedResult,const char* errorMessage)
{
	HRESULT hr;
	m_pQp->Write(&hr,Sge,nSge,remoteAddress,remoteToken,flag);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);

}

void Nd2TestBase::Read(const ND2_SGE* Sge,const ULONG nSge,UINT64 remoteAddress, UINT32 remoteToken,DWORD flag,HRESULT expectedResult,const char* errorMessage)
{
	HRESULT hr;
	m_pQp->Read(&hr,Sge,nSge,remoteAddress,remoteToken,flag);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

void Nd2TestBase::Send(VOID* requestContext,const ND2_SGE* Sge,const ULONG nSge,ULONG flags,HRESULT expectedResult,const char* errorMessage){

	HRESULT hr = m_pQp->Send(requestContext,Sge, nSge,flags);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);

}

void Nd2TestBase::Send(VOID* requestContext,const ND2_SGE* Sge,const ULONG nSge,ULONG flags,bool expectFail,const char* errorMessage){

	HRESULT hr = m_pQp->Send(requestContext,Sge, nSge,flags);

	if(expectFail && !FAILED(hr)){
		LogErrorExit(hr,errorMessage,__LINE__);
	}
	else if(!expectFail && FAILED(hr)){
		LogErrorExit(hr,errorMessage,__LINE__);
	}


}

void Nd2TestBase::WaitForCompletion(HRESULT expectedResult,const char* errorMessage)
{
	HRESULT hr = m_pCq->Notify( ND_CQ_NOTIFY_ANY, &m_Ov );
	ND2_RESULT pResult[1];
	if( hr == ND_PENDING )
	{
		SIZE_T BytesRet;
		hr = m_pCq->GetOverlappedResult( &m_Ov, TRUE );
	}
	m_pCq->GetResults(pResult,1);
	LogIfErrorExit(pResult->Status,expectedResult,errorMessage,__LINE__);
}

void Nd2TestBase::CheckCQ(HRESULT expectedResult,const char* errorMessage)
{
	ND2_RESULT pResult[1];
	int entries = m_pCq->GetResults(pResult,1);
	if(entries != 1){
		LogErrorExit(errorMessage,__LINE__);
	}
	else{
		LogIfErrorExit(pResult->Status,expectedResult,errorMessage,__LINE__);
	}
}
void Nd2TestBase::CheckCQ(int expectedEntries,const char* errorMessage)
{
	ND2_RESULT pResult[1];
	int entries = m_pCq->GetResults(pResult,1);
	if(entries != expectedEntries){
		LogErrorExit(errorMessage,__LINE__);
	}
}

void Nd2TestBase::FlushQP(HRESULT expectedResult,const char* errorMessage)
{
	HRESULT hr = m_pQp->Flush();
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);

}

void Nd2TestBase::Reject(const VOID *pPrivateData,DWORD cbPrivateData, HRESULT expectedResult, const char* errorMessage )
{
	HRESULT hr = m_pConnector->Reject(pPrivateData,cbPrivateData);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}
Nd2TestServerBase::Nd2TestServerBase() :
m_pListen( NULL )
{
}

Nd2TestServerBase::~Nd2TestServerBase()
{
	if( m_pListen != NULL )
	{
		m_pListen->Release();
	}
}

void Nd2TestServerBase::CreateListen(HRESULT expectedResult,const char* errorMessage)
{
	HRESULT hr = m_pAdapter->CreateListener(
		IID_IND2Listener,
		m_hAdapterFile,
		reinterpret_cast<VOID**>( &m_pListen )
		);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}
void Nd2TestServerBase::Listen( _In_ const sockaddr_in& v4Src,HRESULT expectedResult, const char* errorMessage)
{

	HRESULT hr = m_pListen->Bind(
		reinterpret_cast<const sockaddr*>( &v4Src ),
		sizeof( v4Src )
		);
	LogIfErrorExit(hr,expectedResult,"Bind failed",__LINE__);
	hr = m_pListen->Listen(0);
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}
void Nd2TestServerBase::GetConnectionRequest(HRESULT expectedResult, const char* errorMessage)
{
	HRESULT hr = m_pListen->GetConnectionRequest( m_pConnector, &m_Ov );
	if( hr == ND_PENDING )
	{
		hr = m_pListen->GetOverlappedResult( &m_Ov, TRUE );
	}
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

void Nd2TestServerBase::Accept(DWORD inboundReadLimit,DWORD outboundReadLimit,const VOID *pPrivateData,DWORD cbPrivateData,
	HRESULT expectedResult, const char* errorMessage)
{
	//
	// Accept the connection.
	//
	HRESULT hr = m_pConnector->Accept(
		m_pQp,
		inboundReadLimit,
		outboundReadLimit,
		pPrivateData,
		cbPrivateData,
		&m_Ov
		);
	if( hr == ND_PENDING )
	{
		hr = m_pConnector->GetOverlappedResult( &m_Ov, TRUE );
	}
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}



//
// Connect to the server.
//
void Nd2TestClientBase::Connect( _In_ const sockaddr_in& v4Src, _In_ const sockaddr_in& v4Dst,
	DWORD inboundReadLimit,DWORD outboundReadLimit,const VOID *pPrivateData,DWORD cbPrivateData,
	HRESULT expectedResult, const char* errorMessage)
{

	HRESULT hr = m_pConnector->Bind(
		reinterpret_cast<const sockaddr*>( &v4Src ),
		sizeof(v4Src)
		);
	if( hr == ND_PENDING )
	{
		hr = m_pConnector->GetOverlappedResult( &m_Ov, TRUE );
	}

	hr = m_pConnector->Connect(
		m_pQp,
		reinterpret_cast<const sockaddr*>( &v4Dst ),
		sizeof(v4Dst),
		inboundReadLimit,
		outboundReadLimit,
		pPrivateData,
		cbPrivateData,
		&m_Ov
		);
	if( hr == ND_PENDING )
	{
		hr = m_pConnector->GetOverlappedResult( &m_Ov, TRUE );
	}
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

//
// Complete the connection - this transitions the endpoint so it can send.
//
void Nd2TestClientBase::CompleteConnect(HRESULT expectedResult, const char* errorMessage)
{

	HRESULT hr = m_pConnector->CompleteConnect( &m_Ov );
	if( hr == ND_PENDING )
	{
		hr = m_pConnector->GetOverlappedResult( &m_Ov, TRUE );
	}
	LogIfErrorExit(hr,expectedResult,errorMessage,__LINE__);
}

