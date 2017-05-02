#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_REQUEST_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_REQUEST_H_

#include <unordered_map>
#include <cstdint>
#include <memory>
#include "network/rdma/windows/message.h"

namespace oneflow {

class Request {
 public:
  Message* rdma_msg;
  int32_t time_stamp;
  bool is_send;
 private:
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_WINDOWS_REQUEST_H_
