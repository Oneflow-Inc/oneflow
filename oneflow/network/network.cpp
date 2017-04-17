#include "network/network.h"
// #include "network/rdma/rdma_wrapper.h"

#include <vector>
#include <string>
#include <memory>
#include <glog/logging.h>
// #include "common/common.h"

namespace oneflow {

Network* GetNdspiRDMAInstance() {
  // static rdma::RdmaWrapper instance;
  // return &instance;
  LOG(FATAL) << "Unimplemented";
  return nullptr;
}

}  // namespace oneflow
