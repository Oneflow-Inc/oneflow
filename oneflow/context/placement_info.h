#ifndef _CONTEXT_PLACEMENT_INFO_H_
#define _CONTEXT_PLACEMENT_INFO_H_
#include <vector>
#include <unordered_set>
#include <string>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include "caffe.pb.h"
#include <glog/logging.h>
/*
PlacementInfo is a property of an operator(i.e., layer, operator node in DAG) in 
a computation network. It describes on which devices a particular operator will
be executed, and what parallelization policy will be used (i.e., either data-
parallel, or model-parallel).
*/
namespace caffe {
class PlacementInfo {
public:
  PlacementInfo();
  // Use default copy and assign functions
  PlacementInfo(const PlacementInfo& other) = default;
  PlacementInfo& operator=(const PlacementInfo& other) = default;

  const std::vector<int32_t>& device_set() const;
  const std::vector<int32_t>& machine_set() const;
  ParallelPolicy parallel_policy() const;
  bool IsInitialized() const;
  bool EqualTo(const PlacementInfo& other) const;

  void InitWithDeviceGroup(int32_t begin, int32_t end,
    ParallelPolicy parallel_policy, int32_t device_num_per_machine);
  void InitWithMachineGroup(int32_t begin, int32_t end,
    ParallelPolicy parallel_policy);

private:
  std::vector<int32_t> device_set_;
  std::vector<int32_t> machine_set_;
  ParallelPolicy parallel_policy_;

  void SetDeviceSet(
    int32_t begin, int32_t end, int32_t device_num_each_machine);
};

/*
PlacementGroupInfo is an internal representation parsed from PlacementGroup
*/
class PlacementGroupInfo {
public:
  PlacementGroupInfo() = default;
  bool IsInitialized() const;

  const std::string& name() const { return name_; }
  const std::vector<std::string>& layer_set() const { return layer_set_; }
  const PlacementInfo& placement_info() const { return placement_info_; }

  void set_name(const std::string& name) { name_ = name; }
  void add_layer(const std::string& layer) { layer_set_.push_back(layer); }

  void InitPlacementInfoWithDeviceGroup(int32_t begin, int32_t end,
    ParallelPolicy parallel_policy, int32_t device_num_per_machine);
  void InitPlacementInfoWithMachineGroup(int32_t begin, int32_t end,
    ParallelPolicy parallel_policy);
private:
  std::string name_;
  std::vector<std::string> layer_set_;
  PlacementInfo placement_info_;
};
}  // namespace caffe
#endif  // _CONTEXT_PLACEMENT_INFO_H_
