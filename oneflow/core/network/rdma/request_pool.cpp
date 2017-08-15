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
  // There are also some registered_message in |msg_pool_|, however, which will
  // be released in the desctructor of MessagePool.
}

Request* RequestPool::AllocRequest(bool is_send) {
  Request* request = new Request();
  request->is_send = is_send;
  request->rdma_msg = msg_pool_->Alloc();
  if (is_send == false) {
    request->callback = callback4recv_msg_;
    request_vector_.push_back(request);
  }
  return request;
}

void RequestPool::ReleaseRequest(Request* request) {
  // Return the registered message to |msg_pool_|
  msg_pool_->Free(request->rdma_msg);
  // Destroy the Request object
  if (request != nullptr) {
    delete request;
    request = nullptr;
  }
}

}  // namespace oneflow
