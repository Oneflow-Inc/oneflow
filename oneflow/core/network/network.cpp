#include "oneflow/core/network/network.h"

#include <glog/logging.h>
#include <memory>
#include <string>
#include <vector>
// #include "common/common.h"

#include "oneflow/core/network/rdma/rdma_network.h"

namespace oneflow {

Network* GetRdmaInstance() {
  static RdmaNetwork instance;
  return &instance;
  LOG(FATAL) << "Unimplemented";
  return nullptr;
}

}  // namespace oneflow
