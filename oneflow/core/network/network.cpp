#include "oneflow/core/network/network.h"
#include "oneflow/core/network/rdma/rdma_network.h"

namespace oneflow {

Network* GetRdmaInstance() {
  static RdmaNetwork instance;
  return &instance;
}

}  // namespace oneflow
