#include "oneflow/core/network/rdma/windows/rdma_wrapper.h"

#include <Ws2tcpip.h>  // TODO(shiyuan)
#include <windows.h>

#include <string.h>
#include <iostream>
#include "oneflow/core/network/rdma/windows/ndsupport.h"

#include "oneflow/core/network/rdma/windows/interface.h"
#include "oneflow/core/network/rdma/windows/connection.h"
#pragma comment(lib, "Ws2_32.lib")

namespace oneflow {

namespace {

sockaddr_in GetAddress(const char* ip, int port) {
  sockaddr_in addr = sockaddr_in();
  memset(&addr, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, ip, &addr.sin_addr);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  return addr;
}

}  // namespace

RdmaWrapper::RdmaWrapper()
    : adapter_(nullptr),
      send_cq_(nullptr),
      recv_cq_(nullptr),
      listener_(nullptr) {}

RdmaWrapper::~RdmaWrapper() {
  Destroy();
}

void RdmaWrapper::Init(const char* ip, int port) {
  my_addr_ = GetSocket(addr, port);

  // INIT ADAPTER
  // NdspiV2Open
  HRESULT hr = NdStartup();
  CHECK(SUCCEEDED(hr)) << "NdStartup failed. hr = " << hr;
  hr = NdOpenV2Adapter(reinterpret_cast<const sockaddr*>(&my_addr_),
                       sizeof(my_addr_),
                       &adapter_);
  CHECK(SUCCEEDED(hr)) << "Failed to OpenNdV2Adapter, hr = " << hr;
  hr = adapter_->CreateOverlappedFile(&overlapped_file_);
  CHECK(SUCCEEDED(hr));

  ULONG info_size = sizeof(adapter_info_);
  adapter_info_.InfoVersion = ND_VERSION_2;
  hr = adapter_->Query(&adapter_info_, &info_size);
  CHECK(SUCCEEDED(hr));

  // INIT ENV
  // Create Send Completion Queue and Recv Completion Queue
  hr = adapter_->CreateCompletionQueue(
               IID_IND2CompletionQueue,
               overlapped_file_,
               // use max depth as default
               adapter_info_.MaxCompletionQueueDepth,
               0,  // not specify processor group
               0,  // not specify affinity
               reinterpret_cast<void**>(&send_cq_));
  CHECK(SUCCEEDED(hr));

  hr = adapter_->CreateCompletionQueue(
               IID_IND2CompletionQueue,
               overlapped_file_,
               adapter_info_.MaxCompletionQueueDepth,
               0,
               0,
               reinterpret_cast<void**>(&recv_cq_));
  CHECK(SUCCEEDED(hr));

  // StartListen
  hr = adapter_->CreateListener(
               IID_IND2Listener,
               overlapped_file_,
               reinterpret_cast<void**>(&listener_));
  CHECK(SUCCEEDED(hr));

  hr = listener_->Bind(
               reinterpret_cast<const sockaddr*>(&my_sock_),
               sizeof(sockaddr_in));
  CHECK(SUCCEEDED(hr));

  // Start listening for incoming connection requests
  // argument BACKLOG: The maximum number of pending connection requests
  // to maintain for the listen request. Set to zero to indicate no limit.
  hr = listener_->Listen(0);  // NOTE: not sure whether 0(no limit) is OK
  CHECK(SUCCEEDED(hr));
}

void RdmaWrapper::Destroy() {
  // TODO(shiyuan)
}

void RdmaWrapper::CreateConnector(Connection* conn) {
  IND2Connector* connector = nullptr;
  HRESULT hr = adapter_->CreateConnector(
            IID_IND2Connector,
            overlapped_file_,
            reinterpret_cast<void**>(&connector));
  CHECK(SUCCEEDED(hr));
  conn->set_connector(connector);
}

void RdmaWrapper::CreateQueuePair(Connection* conn) {
  IND2QueuePair* queue_pair = nullptr;
  HRESULT hr = adapter_->CreateQueuePair(
            IID_IND2QueuePair,
            recv_cq_,
            send_cq_,
            NULL,
            // just all set them as maximum value, need to be set
            // according to our application protocal carefully.
            adapter_info_.MaxReceiveQueueDepth,
            adapter_info_.MaxInitiatorQueueDepth,
            1,  // adapter_info_.MaxRecvSge,
            adapter_info_.MaxInitiatorSge,
            adapter_info_.MaxInlineDataSize,
            reinterpret_cast<void**>(&queue_pair));
  CHECK(SUCCEEDED(hr));
  conn->set_queue_pair(queue_pair);
}

RdmaMemory* RdmaWrapper::NewNetworkMemory() {
  IND2MemoryRegion* memory_region = nullptr;
  HRESULT hr = adapter_->CreateMemoryRegion(
            IID_IND2MemoryRegion,
            overlapped_file_,
            reinterpret_cast<void**>(&memory_region));
  CHECK(SUCCEEDED(hr));

  RdmaMemory* rdma_memory = new RdmaMemory(memory_region);
  CHECK(rdma_memory);
  return rdma_memory;
}

// XXX(shiyuan)
int64_t RdmaWrapper::WaitForConnection(Connection* conn,
                                       Request* receive_request) {
  int64_t peer_machine_id;
  ULONG size = sizeof(peer_machine_id);

  IND2Connector* connector = conn->connector();
  OVERLAPPED* ov = conn->overlapped();
  HRESULT hr = listener_->GetConnectionRequest(connector, ov);
  if (hr == ND_PENDING) {
    hr = listener_->GetOverlappedResult(ov, TRUE);
  }
  CHECK(SUCCEEDED(hr));

  hr = connector->GetPrivateData(&peer_machine_id, &size);
  // LOG(INFO) << "peer_machine_id = " << peer_machine_id << " size = " << size
  //           << "\n";
  // The author of NDSPI says it's normal for this check failed
  // So just ignore it.
  // CHECK(!FAILED(hr)) << "Failed to get private data. hr = " << hr << "\n";

  conn->set_connector(connector);
  conn->PostRecvRequest(receive_request);
  conn->AcceptConnect();

  return peer_machine_id;
}

// |result| is owned by the caller, and the received message will be held in
// result->net_msg, having result->type == NetworkResultType::kReceiveMsg.
Request* RdmaWrapper::PollRecvQueue(NetworkResult* result) {
  ND2_RESULT nd2_result;
  uint32_t len = recv_cq_->GetResults(&nd2_result, 1);
  if (len == 0)
    return nullptr;

  CHECK_EQ(nd2_result.Status, ND_SUCCESS);

  result->type = NetworkResultType::kReceiveMsg;
  // The context is the message timestamp in Recv Request.
  return reinterpret_cast<Request*>(nd2_result.RequestContext);
}

Request* RdmaWrapper::PollSendQueue(NetworkResult* result) {
  ND2_RESULT nd2_result;
  uint32_t len = send_cq_->GetResults(&nd2_result, 1);
  if (len == 0)
    return nullptr;

  CHECK_EQ(nd2_result.Status, ND_SUCCESS);

  switch (nd2_result.RequestType) {
    case ND2_REQUEST_TYPE::Nd2RequestTypeSend: {
      result->type = NetworkResultType::kSendOk;
      // The context is the message timestamp in Send request.
      // The network object does not have additional information
      // to convey to outside caller, it just recycle the
      // registered_message used in sending out.
      return reinterpret_cast<Request*>(nd2_result.RequestContext);
    }
    case ND2_REQUEST_TYPE::Nd2RequestTypeRead: {
      result->type = NetworkResultType::kReadOk;
      // The context is the message timestamp in Read request.
      // The network object needs to convey the information about
      // "what data have been read" to external caller.
      return reinterpret_cast<Request*>(nd2_result.RequestContext);
    }
    default:
      return nullptr;
  }
}

}  // namespace oneflow
