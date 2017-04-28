#include "rdma_manager.h"

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

} // namespace 

RdmaManager::RdmaManager(const char* addr, int port) {  
  my_sock_ = GetAddress(addr, port);
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
  hr = NdOpenV2Adapter(reinterpret_cast<const sockaddr*>(&my_sock_), 
                       sizeof(my_sock_),
                       &adapter_);
  // CHECK hr

  hr = adapter_->CreateOverlappedFile(&overlapped_file_);
  // CHECK hr
  
  uint64_t info_size = sizeof(adapter_info_);
  adapter_info.InfoVersion = ND_VERSION_2;
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
      adapter_info_.MaxCompletionQueueDepth, // use max depth as default
      0, // not specify processor group
      0, // not specify affinity
      reinterpret_cast<void**>(&send_cq_));
  // CHECK(!FAILED(hr)) << "Failed to create send completion queue\n";

  hr = adapter_->CreateCompletionQueue(
      IID_IND2CompletionQueue,
      overlapped_file_,
      adapter_info_.MaxCompletionQueueDepth.
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
  hr = listener_->Listen(0);  // NOTE(feiga): not sure whether 0(no limit) is OK
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


} // namespace oneflow

