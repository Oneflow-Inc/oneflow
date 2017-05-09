#include "rdma_manager.h"

namespace oneflow {

RdmaManager::RdmaManager() {
  Init();
}

RdmaManager::~RdmaManager() {
  Destroy();
}

bool RdmaManager::Init() {
  return InitDevice() && InitAdapter() && InitEnv();
}

} // namespace oneflow

