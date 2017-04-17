#ifndef _CONTEXT_STRATEGY_DESCRIPTOR_H_
#define _CONTEXT_STRATEGY_DESCRIPTOR_H_
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include "proto/oneflow.pb.h"
#include <glog/logging.h>
#include "context/placement_info.h"
/*
Parsed content from strategy proto.
*/
namespace oneflow {
class ResourceDescriptor;
class StrategyDescriptor {
 public:
  StrategyDescriptor(const oneflow::Strategy& strategy,
    std::shared_ptr<ResourceDescriptor> resource_descriptor);
  ~StrategyDescriptor();

  int32_t group_num() const;
  std::string name(int32_t id) const;
  int32_t layer_num(int32_t id) const;
  std::vector<std::string> layer_set(int32_t id) const;
  int32_t device_num(int32_t id) const;
  std::vector<int32_t> device_set(int32_t id) const;
  std::vector<int32_t> machine_set(int32_t id) const;
  ParallelPolicy parallel_policy(int32_t id) const;
  int32_t group_id_from_name(const std::string& name) const;
  std::string group_from_layer(const std::string& layer_name) const;

  int32_t max_data_parallel_num() const;
  int32_t piece_size_each_device() const;
  int32_t piece_num_each_sync() const;
  int32_t device_num_per_data_provider() const;

  const PlacementGroupInfo& group_info(int32_t id) const;
  const PlacementInfo& placement_info(int32_t id) const;

  void update_placement_info_with_machine_group(
    int32_t id, int32_t begin, int32_t end, ParallelPolicy parallel_policy);
  void update_placement_info_with_device_group(
    int32_t id, int32_t begin, int32_t end, ParallelPolicy parallel_policy);

  bool group_info_is_initialized(int32_t id) const;
  void set_piece_size_each_device(int32_t piece_size_each_device);

 private:
  std::shared_ptr<ResourceDescriptor> resource_descriptor_;
  int32_t group_num_;
  std::vector<PlacementGroupInfo> placement_group_infos_;
  std::unordered_map<std::string, int32_t> name_to_group_ids_;
  std::unordered_map<std::string, std::string> layer_name_to_group_name_;

  // Indicates the maximum degree of data-parallelism. Since currently we require
  // the continuous data-parallelism operators have the same number of devices,
  // the |max_data_parallel_num_| is actually the unique degree of
  // data-parallelism.
  int32_t max_data_parallel_num_{ 1 };
  // Indicates the number of examples processed by the source device in a piece.
  // With "source device", we mean the devices allocated to the operator closest
  // to the data provider node in LogicalDag.
  int32_t piece_size_each_device_{ 1 };
  // For data-parallelism, how many pieces are processed by each device in a
  // synchronization cycle.
  int32_t piece_num_each_sync_{ 1 };
  // The number of devices allocated to the first layer immediately after each
  // data provider.
  int32_t device_num_per_data_provider_{ 1 };

  int32_t pipeline_depth_{ 1 };

  void Init(const oneflow::Strategy& strategy);
  void InitOneGroup(const oneflow::Strategy& strategy, int32_t group_id);
  void ParsePlacementInfo(const PlacementGroup& placement_group,
    PlacementGroupInfo *placement_group_info);
  void HandleDeviceGroup(const PlacementGroup& placement_group,
    PlacementGroupInfo *placement_group_info);
  void HandleMachineGroup(const PlacementGroup& placement_group,
    PlacementGroupInfo *placement_group_info);

  StrategyDescriptor(const StrategyDescriptor& other) = delete;
  StrategyDescriptor& operator=(const StrategyDescriptor& other) = delete;
};
}  // namespace oneflow
#endif  // _CONTEXT_STRATEGY_DESCRIPTOR_H_
