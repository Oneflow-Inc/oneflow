#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

size_t ResourceDesc::TotalMachineNum() const {
  CHECK_GT(resource_.machine_num(), 0);
  CHECK_LE(resource_.machine_num(), Global<EnvDesc>::Get()->TotalMachineNum());
  return resource_.machine_num();
}

const Machine& ResourceDesc::machine(int32_t idx) const {
  CHECK_LT(idx, TotalMachineNum());
  return Global<EnvDesc>::Get()->machine(idx);
}

int32_t ResourceDesc::ComputeThreadPoolSize() const {
  if (resource_.has_compute_thread_pool_size()) {
    CHECK_GT(resource_.compute_thread_pool_size(), 0);
    return resource_.compute_thread_pool_size();
  } else {
    return CpuDeviceNum();
  }
}

bool ResourceDesc::enable_debug_mode() const {
  return std::getenv("ONEFLOW_DEBUG_MODE") != nullptr || resource_.enable_debug_mode();
}

}  // namespace oneflow
