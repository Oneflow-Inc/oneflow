#ifndef ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_
#define ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_

#include "oneflow/core/network/rdma/switch.h"
#include "oneflow/core/network/rdma/message_pool.h"

namespace oneflow {

struct Request {
  RdmaMessage* rdma_msg;
  int32_t time_stamp;
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

  // Release the Request object indexed by |time_stamp|.
  void ReleaseRequest(int32_t time_stamp);

  // Get the Request object indexed by |time_stamp|.
  Request* GetRequest(int32_t time_stamp) const;

  // Update and reuse the Request object indexed by |time_stamp|, to avoid
  // unnecessary object destroy and creation. It is useful when only time_stamp
  // needs update, while other properties do not change.
  Request* UpdateTimeStampAndReuse(int32_t time_stamp);  // TODO(shiyuan)

 private:
  int32_t sequence_number_;
  std::unordered_map<int32_t, Request*> request_dict_;
  std::shared_ptr<MessagePool<RdmaMessage>> msg_pool_;
  static const int32_t kBufferSize = 64;

  int32_t new_time_stamp();

  RequestPool(const RequestPool& other) = delete;
  RequestPool& operator=(const RequestPool& other) = delete;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_REQUEST_POOL_H_
