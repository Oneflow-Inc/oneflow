#include "oneflow/core/network/rdma/windows/rdma_message.h"
#include "oneflow/core/network/rdma/windows/interface.h"

namespace oneflow {

RdmaMessage::RdmaMessage() {
  net_memory_ = dynamic_cast<RdmaMemory*>(
      GetRdmaInstance()->RegisterMemory(
        reinterpret_cast<void*>(&net_msg_), sizeof(net_msg_)));
  net_memory_->Register();
}

RdmaMessage::~RdmaMessage() {
  net_memory_->Unregister();
  delete net_memory_;
}

}  // namespace oneflow

