#include "network/rdma/linux/rdma_message.h"
#include "network/rdma/linux/interface.h"

namespace oneflow {

RdmaMessage::RdmaMessage() {
  net_memory_ = dynamic_cast<RdmaMemory*>(
      GetRdmaInstance()->NewNetworkMemory());
  net_memory_->Reset(reinterpret_cast<char*>(&net_msg_), sizeof(net_msg_));
  net_memory_->Register();
}

RdmaMessage::~RdmaMessage() {
  net_memory_->Unregister();
  delete net_memory_;
}

}  // namespace oneflow
