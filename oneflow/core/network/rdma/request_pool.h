#ifndef ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_
#define ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_

#include <mutex>
#include "oneflow/core/network/rdma/message_pool.h"
#include "oneflow/core/network/rdma/switch.h"

namespace oneflow {

struct Request {
  RdmaMessage* rdma_msg;
  std::function<void(const NetworkMessage& net_msg)> callback;
  bool is_send;
};

class RequestPool {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestPool);
  RequestPool();
  ~RequestPool();

  // Allocate a Request object from the pool.
  // |is_send| is true for Send request, false for Receive request.
  Request* AllocRequest(bool is_send);

  void ReleaseRequest(Request* request);

  void set_callback4recv_msg(
      std::function<void(const NetworkMessage& net_msg)> callback) {
    callback4recv_msg_ = callback;
  }

 private:
  std::mutex mutex_;  // Protect request_vector_
  std::vector<Request*> request_vector_;
  std::unique_ptr<MessagePool<RdmaMessage>> msg_pool_;
  static const int32_t kBufferSize = 64;
  std::function<void(const NetworkMessage& net_msg)> callback4recv_msg_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_
