#include "net/network.h"
//#include "net/rdma/rdma_wrapper.h"

#include <vector>
#include <string>
#include <memory>
#include "common/common.h"

namespace oneflow {

Network* GetNdspiRDMAInstance() {
  //static rdma::RdmaWrapper instance;
  //return &instance;
  Network* network = nullptr;
  return network;
}

}  // namespace caffe
