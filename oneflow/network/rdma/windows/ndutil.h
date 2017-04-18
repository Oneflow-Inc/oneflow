// ndutil.h

#include "ndcommon.h"
#include "ndshim.h"
#include <stdio.h>
#ifndef _ND_UTIL
#define _ND_UTIL

#define RECV    0
#define SEND    1

//logger info
inline void LogIfErrorExit(HRESULT result, HRESULT ExpectedResult,const char* ErrorMessage,int ExitCode)
{
	if(result!=ExpectedResult){
		printf("Test failed.Test Result: %#x, Expected Result:%#x, %s\n",result,ExpectedResult,ErrorMessage);
		exit(ExitCode);
	}
}

inline void LogErrorExit(HRESULT result,const char* ErrorMessage,int ExitCode)
{
	printf("Test failed. Test Result:%#x,%s\n",result,ErrorMessage);
	exit(ExitCode);
}
inline void LogErrorExit(const char* ErrorMessage,int ExitCode)
{
	printf("Test Failed.\n");
	printf("%s\n",ErrorMessage);
	exit(ExitCode);
}
inline void LogIfNoErrorExit(HRESULT result,const char* ErrorMessage,int ExitCode)
{
	if(!FAILED(result))
	{
		printf("Test failed. Test Result:%#x,%s\n",result,ErrorMessage);
		exit(ExitCode);
	}
}

inline void LogInfo(const char* message)
{
	printf("%s\n",message);
}

//base class
class Nd2TestBase
{
protected:
	IND2Adapter* m_pAdapter;
	IND2MemoryRegion* m_pMr;
	IND2CompletionQueue* m_pCq;
	IND2QueuePair* m_pQp;
	IND2Connector* m_pConnector;
	HANDLE m_hAdapterFile;
	DWORD m_Buf_Len;
	void* m_Buf;
	IND2MemoryWindow* m_pMw;
	ND_RESULT m_Results[2];
	OVERLAPPED m_Ov;

protected:
	Nd2TestBase();
	~Nd2TestBase();
	
	//Initialize the adaptor, overlapped handler
	void Init(_In_ const struct sockaddr_in& v4Src);

	//Create a memowy window
	void CreateMW(HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Adapter::CreateMemoryWindow failed");
	
	//invalidate a memory window
	void InvalidateMW(HRESULT expectedResult = ND_SUCCESS, const char* errorMessage="IND2QueuePair:InvalidateMemoryWindow failed");
	
	//create a MR for ND2
	//will reprt error if return value is not same expected
	void CreateMR(HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Adapter::CreateMemoryRegion failed");
	
	//allocate and register data buffer with MR
	void RegisterDataBuffer(DWORD bufferLength,ULONG type,HRESULT expectedResult = ND_SUCCESS,const char* errorMessage = "IND2MemoryRegion::Register failed");
	
	//create copletion queue for given depth
	void CreateCQ(DWORD depth,HRESULT expectedResult = ND_SUCCESS,const char* errorMessage ="IND2Adapter::CreateCompletionQueue failed");
	
	//Create connector with adaptor, must call after init
	void CreateConnector(HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Adapter::CreateConnector failed");
	
	//create Queue pair, only take maxReceiveRequestSge and use the same nSge for both send and receive
	void CreateQueuePair(DWORD queueDepth,DWORD nSge,HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Adapter::CreateQueuePair failed");
	
	//Create Queue pair,allowing all arguments of the CreateQueuePair method
	void CreateQueuePair(DWORD receiveQueueDepth,DWORD initiatorQueueDepth,DWORD maxReceiveRequestSge, DWORD maxInitiatorRequestSge,HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Adapter::CreateQueuePair failed");
	
	//Disconnect Connector and release it
	//No error check
	void DisconnectConnector();

	//Deregister memory
	//No error check
	void DeregisterMemory();

	//Get result from completion queue
	void GetResult(HRESULT expectedResult = ND_SUCCESS,const char* errorMessage = "IND2CompletionQueue::GetOverlappedResult failed");
	
	//Post receive Sges
	void PostReceive(const ND2_SGE* Sge,const DWORD nSge,HRESULT expectedResult = ND_SUCCESS,const char* errorMessage = "IND2QueuePair::Receive failed"); 
	
	//Write to remote peer
	void Write(const ND2_SGE* Sge,const ULONG nSge,UINT64 remoteAddress, UINT32 remoteToken,DWORD flag,HRESULT expectedResult = ND_SUCCESS,const char* errorMessage = "IND2QueuePair::Write failed");
	
	//Send to remote side, does strict error check
	void Send(VOID* requestContext,const ND2_SGE* Sge,const ULONG nSge,ULONG flags,HRESULT expectedResult = ND_SUCCESS,const char* errorMessage = "IND2QueuePair::Send failed");
	
	//Send to remote side, does error check but not very strict
	void Send(VOID* requestContext,const ND2_SGE* Sge,const ULONG nSge,ULONG flags,bool expectFail, const char* errorMessage);
	
	//Wait for overlapped result
	void WaitForCompletion(HRESULT expectedResult = ND_SUCCESS,const char* errorMessage = "Failed");
	
	//bind buffer to MW
	void Bind(DWORD bufferLength,ULONG type,HRESULT expectedResult = ND_SUCCESS, const char* errorMessage ="IND2QueuePair::Bind failed");
	
	//tear down, no error check
	void Shutdown();
	
	//read from remote peer
	void Read(const ND2_SGE* Sge,const ULONG nSge,UINT64 remoteAddress, UINT32 remoteToken,DWORD flag,HRESULT expectedResult=ND_SUCCESS,const char* errorMessage = "IND2QueuePair::Read failed");
	
	//Check 1st item in the completion queue
	void CheckCQ(HRESULT expectedResult,const char* errorMessage = "Completion Queue doesn't have expected number of entries");
	
	//Check # of items in completion queue
	void CheckCQ(int expectedEntries,const char* errorMessage = "Completion Queue doesn't have expected number of entries");
	
	//Flush Queue Pair
	void FlushQP(HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2QueuePair:Flush failed");
	
	//Call Connector::Reject  to reject a connection
	void Reject(const VOID *pPrivateData,DWORD cbPrivateData, HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Connector:Reject failed");
};


class Nd2TestServerBase : public Nd2TestBase
{
protected:
    IND2Listener* m_pListen;

public:
    Nd2TestServerBase();
    ~Nd2TestServerBase();

	//virtual method that each test case must implement
	virtual void RunTest(
        _In_ const struct sockaddr_in& v4Src,
        _In_ DWORD queueDepth,
        _In_ DWORD nSge
        )=0;

	//Create listener
	void CreateListen(HRESULT expectedResult = ND_SUCCESS,const char* errorMessage = "IND2Adapter::CreateListen failed");
    
	//listen to a socket address
	void Listen( _In_ const sockaddr_in& v4Src,HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Listen::Listen failed");
	
	//Get connection request from client
	void GetConnectionRequest(HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Listen::GetConnectionRequest failed");
    
	//Accept the connection request from client
	void Accept(DWORD inboundReadLimit,DWORD outboundReadLimit,const VOID *pPrivateData=NULL,DWORD cbPrivateData=0,
							HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Connector::Accept failed");
};

class Nd2TestClientBase : public Nd2TestBase
{

public:
	//virtual method RunTest, all test cases must implement this method
	virtual void RunTest(
        _In_ const struct sockaddr_in& v4Src,
        _In_ const struct sockaddr_in& v4Dst,
        _In_ DWORD queueDepth,
        _In_ DWORD nSge
        )=0;  


	//connect to server
	//bind then connect
    void Connect( _In_ const sockaddr_in& v4Src, _In_ const sockaddr_in& v4Dst,
								DWORD inboundReadLimit,DWORD outboundReadLimit,const VOID *pPrivateData=NULL,DWORD cbPrivateData=0,
								HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Connector::Connect failed");
	
	// Complete the connection - this transitions the endpoint so it can send.
	void CompleteConnect(HRESULT expectedResult = ND_SUCCESS, const char* errorMessage = "IND2Connector::CompleteConnect failed");
};

#endif
