#include "context/strategy_descriptor.h"
#include <glog/logging.h>
#include "caffe.pb.h"
#include "proto_io.h"
#include "context/resource_descriptor.h"
#include <string>
#include <vector>

namespace caffe {
StrategyDescriptor::StrategyDescriptor(const caffe::Strategy& strategy,
  std::shared_ptr<ResourceDescriptor> resource_descriptor)
  : resource_descriptor_(resource_descriptor) {
  Init(strategy);
}

StrategyDescriptor::~StrategyDescriptor() {
}

int32_t StrategyDescriptor::group_num() const {
  return group_num_;
}

std::string StrategyDescriptor::name(int32_t id) const {
  return placement_group_infos_[id].name();
}

int32_t StrategyDescriptor::layer_num(int32_t id) const {
  return placement_group_infos_[id].layer_set().size();
}

std::vector<std::string> StrategyDescriptor::layer_set(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, group_num_);
  return placement_group_infos_[id].layer_set();
}

int32_t StrategyDescriptor::device_num(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, group_num_);
  return placement_group_infos_[id].placement_info().device_set().size();
}

std::vector<int32_t> StrategyDescriptor::device_set(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, group_num_);
  return placement_group_infos_[id].placement_info().device_set();
}

std::vector<int32_t> StrategyDescriptor::machine_set(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, group_num_);
  return placement_group_infos_[id].placement_info().machine_set();
}

ParallelPolicy StrategyDescriptor::parallel_policy(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, group_num_);
  return placement_group_infos_[id].placement_info().parallel_policy();
}

int32_t StrategyDescriptor::group_id_from_name(
  const std::string& name) const {
  auto it = name_to_group_ids_.find(name);
  CHECK(it != name_to_group_ids_.end())
    << "No group with the name: " << name;
  return it->second;
}

std::string StrategyDescriptor::group_from_layer(
  const std::string& layer_name) const {
  auto it = layer_name_to_group_name_.find(layer_name);
  CHECK(it != layer_name_to_group_name_.end());
  return it->second;
}

int32_t StrategyDescriptor::max_data_parallel_num() const {
  return max_data_parallel_num_;
}

int32_t StrategyDescriptor::piece_size_each_device() const {
  return piece_size_each_device_;
}

int32_t StrategyDescriptor::piece_num_each_sync() const {
  return piece_num_each_sync_;
}

int32_t StrategyDescriptor::device_num_per_data_provider() const {
  return device_num_per_data_provider_;
}

const PlacementGroupInfo& StrategyDescriptor::group_info(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, group_num_);
  return placement_group_infos_[id];
}

const PlacementInfo& StrategyDescriptor::placement_info(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, group_num_);
  return placement_group_infos_[id].placement_info();
}

bool StrategyDescriptor::group_info_is_initialized(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, group_num_);
  return placement_group_infos_[id].IsInitialized();
}

void StrategyDescriptor::set_piece_size_each_device(
  int32_t piece_size_each_device) {
  piece_size_each_device_ = piece_size_each_device;
}

void StrategyDescriptor::Init(const caffe::Strategy& strategy) {
  group_num_ = strategy.placement_group_size();
  for (int32_t gid = 0; gid < group_num_; ++gid) {
    InitOneGroup(strategy, gid);
  }
  // Update the device_num_per_data_provider_
  device_num_per_data_provider_
    = max_data_parallel_num_ / resource_descriptor_->machine_num();
}

void StrategyDescriptor::InitOneGroup(
  const caffe::Strategy& strategy, int32_t group_id) {
  auto& placement_group = strategy.placement_group(group_id);
  PlacementGroupInfo placement_group_info;
  placement_group_info.set_name(placement_group.name());
  // Set the layers in this group
  int32_t layer_num = placement_group.layer_set().name_size();
  for (int32_t lid = 0; lid < layer_num; ++lid) {
    auto layer_name = placement_group.layer_set().name(lid);
    placement_group_info.add_layer(layer_name);
    CHECK(layer_name_to_group_name_.count(layer_name) == 0);
    layer_name_to_group_name_.insert({ layer_name, placement_group_info.name()});
  }
  // Parse the placemeng_info
  ParsePlacementInfo(placement_group, &placement_group_info);

  placement_group_infos_.push_back(placement_group_info);
  auto it = name_to_group_ids_.find(placement_group_info.name());
  CHECK(it == name_to_group_ids_.end()) << "Duplicate group names";
  name_to_group_ids_.insert({ placement_group_info.name(), group_id });
}

void StrategyDescriptor::ParsePlacementInfo(
  const PlacementGroup& placement_group,
  PlacementGroupInfo* placement_group_info) {
  //int32_t device_num_each_machine
  //  = resource_descriptor_->device_num_per_machine();
  CHECK(
    (placement_group.has_device_group()
    && placement_group.has_parallel_policy())
    || (placement_group.has_machine_group()
    && placement_group.has_parallel_policy())
    || (!placement_group.has_device_group()
    && !placement_group.has_parallel_policy())
    )
    << "Strategy proto should be one of the following setting: "
    << "(1) device_group and parallel_policy; "
    << "(2) machine_group and parallel_policy; "
    << "(3) empty";
  CHECK(!(placement_group.has_device_group()
    && placement_group.has_machine_group()))
    << "Do not set device_group and machine_group at the same time";

  if (placement_group.has_device_group()
    && placement_group.has_parallel_policy()) {
    HandleDeviceGroup(placement_group, placement_group_info);
  }

  if (placement_group.has_machine_group()
    && placement_group.has_parallel_policy()) {
    HandleMachineGroup(placement_group, placement_group_info);
  }
}

void StrategyDescriptor::HandleDeviceGroup(const PlacementGroup& placement_group,
  PlacementGroupInfo* placement_group_info) {
  int32_t begin = placement_group.device_group().begin();
  int32_t end = placement_group.device_group().end();

  placement_group_info->InitPlacementInfoWithDeviceGroup(begin, end,
    placement_group.parallel_policy(),
    resource_descriptor_->device_num_per_machine());

  // Update |max_data_parallel_num_|
  if (placement_group_info->placement_info().parallel_policy()
    == kDataParallelOnMultipleDevices) {
    CHECK(placement_group_info->placement_info().device_set().size() >= 1)
      << "There must be at least one device for kDataParallelOnMultipleDevices";
    if (placement_group_info->placement_info().device_set().size() >
      max_data_parallel_num_) {
      max_data_parallel_num_
        = placement_group_info->placement_info().device_set().size();
    }
  }
}

void StrategyDescriptor::HandleMachineGroup(const PlacementGroup& placement_group,
  PlacementGroupInfo* placement_group_info) {
  int32_t begin = placement_group.machine_group().begin();
  int32_t end = placement_group.machine_group().end();

  placement_group_info->InitPlacementInfoWithMachineGroup(begin, end,
    placement_group.parallel_policy());
}

void StrategyDescriptor::update_placement_info_with_machine_group(int32_t id,
  int32_t begin, int32_t end, ParallelPolicy parallel_policy) {
  placement_group_infos_[id].InitPlacementInfoWithMachineGroup(
    begin, end, parallel_policy);
}

void StrategyDescriptor::update_placement_info_with_device_group(int32_t id,
  int32_t begin, int32_t end, ParallelPolicy parallel_policy) {
  placement_group_infos_[id].InitPlacementInfoWithDeviceGroup(
    begin, end, parallel_policy, resource_descriptor_->device_num_per_machine());
}

}  // namespace caffe
