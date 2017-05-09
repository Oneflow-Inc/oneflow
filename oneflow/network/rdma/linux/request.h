#ifndef ONEFLOW_NETWORK_RDMA_LINUX_REQUEST_H_
#define ONEFLOW_NETWORK_RDMA_LINUX_REQUEST_H_

#include <unordered_map>
#include <cstdint>
#include <memory>
//#include "network/rdma/message_pool.h"
#include "network/rdma/linux/message.h"

namespace oneflow {
class Request {
public:

private:
    bool is_send_;
    int32_t time_stamp_;
    Message* rdma_msg_;
};
} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_LINUX_REQUEST_H_
