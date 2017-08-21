#include "oneflow/core/network/rdma/request_pool.h"

namespace oneflow {

RequestPool::RequestPool() {
  msg_pool_.reset(new MessagePool<RdmaMessage>(kBufferSize));
}

RequestPool::~RequestPool() {
  for (auto& elemt : request_vector_) {
    CHECK(elemt);
    ReleaseRequest(elemt);
  }
}

Request* RequestPool::AllocRequest(bool is_send) {
  Request* request = new Request();
  request->is_send = is_send;
  request->rdma_msg = msg_pool_->Alloc();
  if (is_send == false) {
    request->callback = callback4recv_msg_;

    std::lock_guard<std::mutex> lock(mutex_);
    request_vector_.push_back(request);
  }
  return request;
}

void RequestPool::ReleaseRequest(Request* request) {
  msg_pool_->Free(request->rdma_msg);
  if (request != nullptr) {
    delete request;
    request = nullptr;
  }
}

}  // namespace oneflow
