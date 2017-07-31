#include "oneflow/core/network/rdma/linux/rdma_message.h"
#include "oneflow/core/network/rdma/linux/interface.h"

namespace oneflow {

RdmaMessage::RdmaMessage() {
  net_memory_ =
      dynamic_cast<RdmaMemory*>(GetRdmaInstance()->RegisterMemory(
          reinterpret_cast<void*>(&net_msg_), sizeof(net_msg_)));
  CHECK(net_memory_);
  net_memory_->Register();
}

RdmaMessage::~RdmaMessage() {
  net_memory_->Unregister();
  delete net_memory_;
  net_memory_ = nullptr;
}

}  // namespace oneflow
