#include "oneflow/core/network/rdma/request_pool.h"

namespace oneflow {

RequestPool::RequestPool() : sequence_number_(0) {
  msg_pool_.reset(new MessagePool<RdmaMessage>(kBufferSize));
}

RequestPool::~RequestPool() {
  for (auto& pair : request_dict_) {
    // Firstly, release the |registered_message| of this request.
    delete pair.second->rdma_msg;
    pair.second->rdma_msg = nullptr;
    // Secondly, release the request itself.
    delete pair.second;
  }
  // There are also some registered_message in |msg_pool_|, however, which will
  // be released in the desctructor of MessagePool.
}

Request* RequestPool::AllocRequest(bool is_send) {
  Request* request = new Request();
  request->is_send = is_send;
  if (is_send == false) {
    request->callback = callback4recv_msg_;
  }
  request->rdma_msg = msg_pool_->Alloc();
  request_set_.insert(request);  // TODO(shiyuan)
  return request;
}

void RequestPool::ReleaseRequest(Request* request) {
  // Return the registered message to |msg_pool_|
  msg_pool_->Free(request->rdma_msg);
  // Destroy the Request object
  delete request;
}

}  // namespace oneflow
