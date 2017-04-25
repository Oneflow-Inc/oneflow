#include "rdma_manager.h"

namespace oneflow {

RdmaManager::RdmaManager() {}

RdmaManager::~RdmaManager() {
  Destroy();
}

bool RdmaManager::Init() {
  return InitAdapter() && InitEnv();
}

bool RdmaManager::InitAdapter() {
  HRESULT hr = NdStartup();

  hr = NdOpenV2Adapter(reinterpret_cast<const sockaddr*>(&sin), 
                       sizeof(sin), 
                       &adapter_);
  // CHECK

  hr = adapter_->CreateOverlappedFile(&overlapped_file_);
  // CHECK
  
  uint64_t info_size = sizeof(adapter_info_);
  adapter_info.InfoVersion = ND_VERSION_2;
  hr = adapter_->Query(&adapter_info_, &info_size);
  // CHECK
  return true;
}

bool RdmaManager::InitEnv() {
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

  hr = adapter_->CreateListener(
      IID_IND2Listener,
      overlapped_file_,
      reinterpret_cast<void**>(&listener_));
  // CHECK(!FAILED(hr)) << "Failed to create listener\n";

  sockaddr_in my_sock = sin;

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

bool RdmaManager::Destroy() {
  delete adapter_;
  delete adapter_info_;
  delete sin;
  delete overlapped_file_;
  delete listener_;
  delete send_cq_;
  delete recv_cq_;

  return true;
}


} // namespace oneflow

