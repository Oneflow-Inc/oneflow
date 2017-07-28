#ifndef ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_
#define ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_

#include "oneflow/core/network/rdma/switch.h"
#include "oneflow/core/network/rdma/message_pool.h"

namespace oneflow {

struct Request {
  RdmaMessage* rdma_msg;
  std::function<void()> callback;
  bool is_send;
};

class RequestPool {
// TODO(shiyuan): need refine
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestPool);
  RequestPool();
  ~RequestPool();

  // Allocate a Request object from the pool.
  // |is_send| is true for Send request, false for Receive request.
  Request* AllocRequest(bool is_send);

  void ReleaseRequest(Request* request);

  // Update and reuse the Request object indexed by |time_stamp|, to avoid
  // unnecessary object destroy and creation. It is useful when only time_stamp
  // needs update, while other properties do not change.
  void ReuseRequest(Request* request);
  
  void set_callback4recv_msg(std::function<void(const ActorMsg&)>) {
    callback4recv_msg_ = callback;
  }
  std::function<void(const ActorMsg&)> callback4recv_msg() {
    return callback4recv_msg_;
  }

 private:
  int32_t used_request_number_;
  int32_t total_request_number_;
  std::unordered_map<int32_t, Request*> request_dict_;
  std::shared_ptr<MessagePool<RdmaMessage>> msg_pool_;
  static const int32_t kBufferSize = 64;
  std::function<void()> callback4recv_msg_;  // TODO(shiyuan): heap or stack?

  RequestPool(const RequestPool& other) = delete;
  RequestPool& operator=(const RequestPool& other) = delete;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_
