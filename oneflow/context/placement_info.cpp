#include "context/placement_info.h"
#include "common/stl_util.h"

namespace oneflow {
PlacementInfo::PlacementInfo() : parallel_policy_(kUnknownParallel) {
}

bool PlacementInfo::IsInitialized() const {
  return parallel_policy_ != kUnknownParallel;
}
void PlacementInfo::SetDeviceSet(
  int32_t begin,
  int32_t end,
  int32_t device_num_each_machine) {
  CHECK_GE(begin, 0);
  CHECK_GE(end, begin);
  device_set_.clear();
  machine_set_.clear();
  for (; begin <= end; ++begin) {
    device_set_.push_back(begin);
    // We ensure the machine_set_ is always consistent with the device_set_
    if (begin % device_num_each_machine == 0) {
      machine_set_.push_back(begin / device_num_each_machine);
    }
  }
}

bool PlacementInfo::EqualTo(const PlacementInfo& other) const {
  // No need to verify the equality of machine_set_
  if (parallel_policy_ == other.parallel_policy()
    && stl::VectorEqual(device_set_, other.device_set())) {
    return true;
  } else {
    return false;
  }
}

const std::vector<int32_t>& PlacementInfo::device_set() const {
  return device_set_;
}

const std::vector<int32_t>& PlacementInfo::machine_set() const {
  return machine_set_;
}

ParallelPolicy PlacementInfo::parallel_policy() const {
  return parallel_policy_;
}

void PlacementInfo::InitWithDeviceGroup(int32_t begin, int32_t end,
  ParallelPolicy parallel_policy, int32_t device_num_per_machine) {
  parallel_policy_ = parallel_policy;
  if (end == begin) {
    CHECK(parallel_policy_ == kNaiveParallelOnSingleDevice)
      << "Single device needs kNaiveParallelOnSingleDevice";
  } else {
    CHECK(parallel_policy_ == kDataParallelOnMultipleDevices
      || parallel_policy_ == kModelParallelOnMultipleDevices)
      << "Multiple devices, either data-parallel or model-parallel";
  }
  SetDeviceSet(begin, end, device_num_per_machine);
}

void PlacementInfo::InitWithMachineGroup(int32_t begin, int32_t end,
  ParallelPolicy parallel_policy) {
  parallel_policy_ = parallel_policy;
  if (end == begin) {
    CHECK(parallel_policy_ == kNaiveParallelOnSingleMachine)
      << "Single host needs kNaiveParallelOnSingleMachine";
  } else {
    CHECK(parallel_policy_ == kDataParallelOnMultipleMachines
      || parallel_policy_ == kModelParallelOnMultipleMachines)
      << "Multiple hosts, either data-parallel or model-parallel";
  }
  CHECK_LE(begin, end);
  for (; begin <= end; ++begin) {
    machine_set_.push_back(begin);
  }
}

bool PlacementGroupInfo::IsInitialized() const {
  return placement_info_.IsInitialized();
}

void PlacementGroupInfo::InitPlacementInfoWithDeviceGroup(
  int32_t begin, int32_t end,
  ParallelPolicy parallel_policy, int32_t device_num_per_machine) {
  placement_info_.InitWithDeviceGroup(begin, end, parallel_policy,
    device_num_per_machine);
}

void PlacementGroupInfo::InitPlacementInfoWithMachineGroup(
  int32_t begin, int32_t end,
  ParallelPolicy parallel_policy) {
  placement_info_.InitWithMachineGroup(begin, end, parallel_policy);
}

}  // namespace oneflow
