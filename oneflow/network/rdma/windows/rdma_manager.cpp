#include "network/rdma/windows/rdma_manager.h"

#include <string.h>
#include "network/rdma/windows/ndsupport.h"
#include "ndsupport.h"

#include "network/rdma/windows/interface.h"
#include "network/rdma/windows/connection.h"

namespace oneflow {

namespace {

sockaddr_in GetAddress(const char* addr, int port) {
  sockaddr_in sock = sockaddr_in();
  std::memset(&sock, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, addr, &sock.sin_addr);
  sock.sin_family = AF_INET;
  sock.sin_port = htons(static_cast<u_short>(port));
  return sock;
}

}  // namespace


RdmaManager::RdmaManager(const char* addr, int32_t port) {
  my_sock = GetAddress(addr, port);
  adapter_ = NULL;
  listener_ = NULL;
  send_cq_ = NULL;
  recv_cq_ = NULL;
}

RdmaManager::~RdmaManager() {
  Destroy();
}

bool RdmaManager::Init() {
  return InitAdapter() && InitEnv();
}

bool RdmaManager::InitAdapter() {
  // NdspiV2Open
  HRESULT hr = NdStartup();
  // CHECK hr
  hr = NdOpenV2Adapter(reinterpret_cast<const sockaddr*>(&my_sock),
                       sizeof(my_sock),
                       &adapter_);
  // CHECK hr

  hr = adapter_->CreateOverlappedFile(&overlapped_file_);
  // CHECK hr

  ULONG info_size = sizeof(adapter_info_);
  adapter_info_.InfoVersion = ND_VERSION_2;
  hr = adapter_->Query(&adapter_info_, &info_size);
  // CHECK hr
  return true;
}

bool RdmaManager::InitEnv() {
  // Create Send Completion Queue and Recv Completion Queue
  HRESULT hr;
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
      reinterpret_cast<const sockaddr*>(&my_sock),
      sizeof(sockaddr_in));
  // CHECK(!FAILED(hr)) << "Failed to bind\n";

  // Start listening for incoming connection requests
  // argument BACKLOG: The maximum number of pending connection requests
  // to maintain for the listen request. Set to zero to indicate no limit.
  hr = listener_->Listen(0);  // NOTE(feiga): not sure whether 0(no limit) is OK
  // CHECK(!FAILED(hr)) << "Failed to Listen\n";

  return true;
}

bool RdmaManager::CreateConnector(Connection* conn) {
  return adapter_->CreateConnector(
      IID_IND2Connector,
      overlapped_file_,
      reinterpret_cast<void**>(&conn->connector));
}

bool RdmaManager::CreateQueuePair(Connection* conn) {
  return adapter_->CreateQueuePair(
      IID_IND2Connector,
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
      reinterpret_cast<void**>(&conn->queue_pair));
}

// FIXME(shiyuan) bug
uint64_t RdmaManager::WaitForConnection(Connection* conn) {
  uint64_t peer_machine_id;
  ULONG size = sizeof(peer_machine_id);
  HRESULT hr = listener_->GetConnectionRequest(conn->connector, &conn->ov);
  if (hr == ND_PENDING) {
    hr = listener_->GetOverlappedResult(&conn->ov, true);
  }
  // CHECK(!FAILED(hr)) << "Failed to GetConnectionRequest\n";
  // LOG(INFO) << "Get connection request done\n";
  //

  // Get src rank from the private data
  hr = conn->connector->GetPrivateData(&peer_machine_id, &size);
  // LOG(INFO) << "peer_machine_id = " << peer_machine_id << " size = " << size << "\n";
  // NOTE(feiga): The author of NDSPI says it's normal for this check failed
  //              So just ignore it.
  // CHECK(!FAILED(hr)) << "Failed to get private data. hr = " << hr << "\n";

  return peer_machine_id;
}

Memory* RdmaManager::NewNetworkMemory() {
  IND2MemoryRegion* memory_region = NULL;
  HRESULT hr = adapter_->CreateMemoryRegion(
      IID_IND2MemoryRegion,
      overlapped_file_,
      reinterpret_cast<void**>(&memory_region));

  Memory* memory = new Memory(memory_region);
  return memory;
}

// |result| is owned by the caller, and the received message will be held in
// result->net_msg, having result->type == NetworkResultType::NET_RECEIVE_MSG.
int32_t RdmaManager::PollRecvQueue(NetworkResult* result) {
  ND2_RESULT nd2_result;
  uint32_t len = recv_cq_->GetResults(&nd2_result, 1);
  if (len == 0)
    return -1;

  // CHECK
  // CHECK

  result->type = NetworkResultType::NET_RECEIVE_MSG;
  // The context is the message timestamp in Recv Request.
  int32_t time_stamp = *(static_cast<int32_t*>(nd2_result.RequestContext));
  return time_stamp;
}

// TODO(shiyuan) should mv PollSendQueue to Class RdmaManager
int32_t RdmaManager::PollSendQueue(NetworkResult* result) {
  // CHECK result
  // HRESULT hr; // FIXME(shiyuan)
  ND2_RESULT nd2_result;
  uint32_t len = send_cq_->GetResults(&nd2_result, 1);
  if (len == 0)
    return -1;

  // CHECK

  // NET_SEND_OK? NET_SEND_ACK?
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
  }
}

bool RdmaManager::Destroy() {
  send_cq_->Release();
  recv_cq_->Release();
  listener_->Release();
  adapter_->Release();
  return true;
}

}  // namespace oneflow
