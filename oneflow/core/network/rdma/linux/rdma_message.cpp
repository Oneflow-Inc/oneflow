#include "oneflow/core/network/rdma/linux/rdma_message.h"

namespace oneflow {

RdmaMessage::RdmaMessage() {
  net_memory_ =
      dynamic_cast<RdmaMemory*>(GetRdmaInstance()->RegisterMemory(
          reinterpret_cast<void*>(&net_msg_), sizeof(net_msg_)));
  CHECK(net_memory_);
}

RdmaMessage::~RdmaMessage() {
  net_memory_->Unregister();
}

}  // namespace oneflow
