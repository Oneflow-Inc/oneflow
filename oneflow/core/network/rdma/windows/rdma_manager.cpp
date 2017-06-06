#include "oneflow/core/network/rdma/windows/rdma_manager.h"

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

sockaddr_in GetSocket(const char* addr, int port) {
  sockaddr_in sock = sockaddr_in();
  memset(&sock, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, addr, &sock.sin_addr);
  sock.sin_family = AF_INET;
  sock.sin_port = htons(static_cast<u_short>(port));
  return sock;
}

}  // namespace

RdmaManager::RdmaManager() {
  adapter_ = NULL;
  listener_ = NULL;
  send_cq_ = NULL;
  recv_cq_ = NULL;
}

RdmaManager::~RdmaManager() {
  Destroy();
}

bool RdmaManager::Init(const char* addr, int port) {
  my_sock_ = GetSocket(addr, port);

  // INIT ADAPTER
  // NdspiV2Open
  HRESULT hr = NdStartup();
  // CHECK hr
  hr = NdOpenV2Adapter(reinterpret_cast<const sockaddr*>(&my_sock_),
                       sizeof(my_sock_),
                       &adapter_);
  // CHECK hr

  hr = adapter_->CreateOverlappedFile(&overlapped_file_);
  // CHECK hr

  ULONG info_size = sizeof(adapter_info_);
  adapter_info_.InfoVersion = ND_VERSION_2;
  hr = adapter_->Query(&adapter_info_, &info_size);
  // CHECK hr

  // INIT ENV
  // Create Send Completion Queue and Recv Completion Queue
  hr = adapter_->CreateCompletionQueue(
      IID_IND2CompletionQueue,
      overlapped_file_,
      adapter_info_.MaxCompletionQueueDepth,  // use max depth as default
      0,  // not specify processor group
      0,  // not specify affinity
      reinterpret_cast<void**>(&send_cq_));
  // CHECK(!FAILED(hr)) << "Failed to create send completion queue\n";

  hr = adapter_->CreateCompletionQueue(
      IID_IND2CompletionQueue,
      overlapped_file_,
      adapter_info_.MaxCompletionQueueDepth,
      0,
      0,
      reinterpret_cast<void**>(&recv_cq_));
  // CHECK(!FAILED(hr)) << "Failed to create recv completion queue\n";

  // StartListen
  hr = adapter_->CreateListener(
      IID_IND2Listener,
      overlapped_file_,
      reinterpret_cast<void**>(&listener_));
  // CHECK(!FAILED(hr)) << "Failed to create listener\n";

  hr = listener_->Bind(
      reinterpret_cast<const sockaddr*>(&my_sock_),
      sizeof(sockaddr_in));
  // CHECK(!FAILED(hr)) << "Failed to bind\n";

  // Start listening for incoming connection requests
  // argument BACKLOG: The maximum number of pending connection requests
  // to maintain for the listen request. Set to zero to indicate no limit.
  hr = listener_->Listen(0);  // NOTE: not sure whether 0(no limit) is OK
  // CHECK(!FAILED(hr)) << "Failed to Listen\n";

  return true;
}

bool RdmaManager::Destroy() {
  send_cq_->Release();
  recv_cq_->Release();
  listener_->Release();
  adapter_->Release();
  return true;
}

bool RdmaManager::CreateConnector(Connection* conn) {
  IND2Connector* connector = NULL;
  HRESULT hr = adapter_->CreateConnector(
      IID_IND2Connector,
      overlapped_file_,
      reinterpret_cast<void**>(&connector));
  conn->set_connector(connector);

  if (SUCCEEDED(hr)) {
    return true;
  } else {
    return false;
  }
}

bool RdmaManager::CreateProtectDomain(Connection* conn) {
  return true;
}

bool RdmaManager::CreateQueuePair(Connection* conn) {
  IND2QueuePair* queue_pair = NULL;
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
  conn->set_queue_pair(queue_pair);

  if (SUCCEEDED(hr)) {
    return true;
  } else {
    return false;
  }
}

RdmaMemory* RdmaManager::NewNetworkMemory() {
  IND2MemoryRegion* memory_region = NULL;
  HRESULT hr = adapter_->CreateMemoryRegion(
      IID_IND2MemoryRegion,
      overlapped_file_,
      reinterpret_cast<void**>(&memory_region));

  RdmaMemory* rdma_memory = new RdmaMemory(memory_region);
  return rdma_memory;
}

// FIXME(shiyuan) bug
uint64_t RdmaManager::WaitForConnection(Connection* conn,
                                        Request* receive_request) {
  uint64_t peer_machine_id;
  ULONG size = sizeof(peer_machine_id);

  IND2Connector* connector = conn->connector();
  OVERLAPPED* ov = conn->overlapped();
  HRESULT hr = listener_->GetConnectionRequest(connector, ov);
  if (hr == ND_PENDING) {
    hr = listener_->GetOverlappedResult(ov, TRUE);
  }
  // CHECK(!FAILED(hr)) << "Failed to GetConnectionRequest\n";
  // LOG(INFO) << "Get connection request done\n";
  //
  if (FAILED(hr)) {
    std::cout << "Failed to GetConnectionRequest" << std::endl;
  }

  if (SUCCEEDED(hr)) {
    connector->GetPrivateData(&peer_machine_id, &size);
    conn->set_connector(connector);
    conn->PostRecvRequest(receive_request);
    std::cout << "Get peer_machine_id = " << peer_machine_id << std::endl;
    conn->AcceptConnect();
  }

  // LOG(INFO) << "peer_machine_id = " << peer_machine_id << " size = " << size
  //           << "\n";
  // The author of NDSPI says it's normal for this check failed
  // So just ignore it.
  // CHECK(!FAILED(hr)) << "Failed to get private data. hr = " << hr << "\n";
  return peer_machine_id;
}

// |result| is owned by the caller, and the received message will be held in
// result->net_msg, having result->type == NetworkResultType::NET_RECEIVE_MSG.
int32_t RdmaManager::PollRecvQueue(NetworkResult* result) {
  ND2_RESULT nd2_result;
  uint32_t len = recv_cq_->GetResults(&nd2_result, 1);
  if (len == 0)
    return -1;

  if (nd2_result.Status != ND_SUCCESS)
    return -1;

  // CHECK
  // CHECK

  result->type = NetworkResultType::NET_RECEIVE_MSG;
  // The context is the message timestamp in Recv Request.
  int32_t time_stamp = *(static_cast<int32_t*>(nd2_result.RequestContext));
  return time_stamp;
}

int32_t RdmaManager::PollSendQueue(NetworkResult* result) {
  ND2_RESULT nd2_result;
  uint32_t len = send_cq_->GetResults(&nd2_result, 1);
  if (len == 0)
    return -1;

  switch (nd2_result.RequestType) {
    case ND2_REQUEST_TYPE::Nd2RequestTypeSend: {
      result->type = NetworkResultType::NET_SEND_OK;
      // The context is the message timestamp in Send request.
      // The network object does not have additional information
      // to convey to outside caller, it just recycle the
      // registered_message used in sending out.
      int32_t time_stamp = *(static_cast<int32_t*>(nd2_result.RequestContext));
      return time_stamp;
    }
    case ND2_REQUEST_TYPE::Nd2RequestTypeRead: {
      result->type = NetworkResultType::NET_READ_OK;
      // The context is the message timestamp in Read request.
      // The network object needs to convey the information about
      // "what data have been read" to external caller.
      int32_t time_stamp = *(static_cast<int32_t*>(nd2_result.RequestContext));
      return time_stamp;
    }
    default:
      return -1;
  }
}

}  // namespace oneflow
