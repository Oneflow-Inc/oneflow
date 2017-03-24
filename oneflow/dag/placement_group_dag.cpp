#include "dag/placement_group_dag.h"
#include <unordered_set>
#include <stack>
#include <limits>
#include "common/common.h"
#include "common/stl_util.h"
#include "dag/node_meta.h"
#include "dag/dag_iterator.h"
#include "dag/logical_dag.h"
#include "context/one.h"
#include "context/config_parser.h"
#include "context/strategy_descriptor.h"
#include "layers/base_layer.h"

namespace oneflow {
template <typename Dtype>
PlacementGroupDag<Dtype>::PlacementGroupDag(
  std::shared_ptr<LogicalDag<Dtype>> logical_dag,
  std::shared_ptr<StrategyDescriptor> strategy_descriptor,
  PathType path_type,
  const std::string& name = "placement_group_dag")
  : Dag(path_type, name), strategy_descriptor_(strategy_descriptor),
  logical_dag_(logical_dag) {
  VerifyPreConditions();
  Build();
  PrintDag(name_);
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::VerifyPreConditions() {
  VerifyDependency();
  CompleteGroupInfo();
  SetPlacementInfoForLayer();
  VerifyElemWiseLayerDependency();
  // VerifyGroupInModelParallel();
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::VerifyDependency() {
  std::unordered_set<std::string> available_layers;
  std::unordered_set<std::string> all_layer_set;
  int32_t group_num = strategy_descriptor_->group_num();
  for (int32_t gid = 0; gid < group_num; ++gid) {
    auto group_name = strategy_descriptor_->name(gid);
    auto layer_set = strategy_descriptor_->layer_set(gid);
    auto layer_num = strategy_descriptor_->layer_num(gid);
    // Each PlacementGroup at least has one layer included.
    CHECK(layer_num > 0);
    for (int32_t layer_id = 0; layer_id < layer_num; ++layer_id) {
      auto layer_name = layer_set[layer_id];
      // Each layer can only belong to a single PlacementGroup.
      CHECK(all_layer_set.count(layer_name) == 0);
      all_layer_set.insert(layer_name);

      // The layer name must be consistent with that of LogicalDag.
      auto layer_ptr = logical_dag_->GetOpNode(layer_name);
      CHECK(layer_ptr != nullptr);

      auto predecessors = logical_dag_->GetPrecedingOpNodeNames(layer_name);

      // Ensure the PlacementGroups are declared in topological order.
      for (auto& predecessor : predecessors) {
        CHECK(available_layers.count(predecessor) > 0);
      }
      available_layers.insert(layer_name);
    }
  }
  // Ensure every layer in LogicalDag is declared in PlacementGroups
  DagIterator<LogicalDag<Dtype>, true> dag_iterator(*logical_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto current_name = current_node->node_name();
    CHECK(all_layer_set.count(current_name) > 0);
  }
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::CompleteGroupInfoForwardDependency(int32_t gid) {
  // Get all the layers who succeed the layers of group |gid|
  auto layer_set = strategy_descriptor_->layer_set(gid);
  auto layer_num = strategy_descriptor_->layer_num(gid);
  std::vector<std::string> successors;
  for (int32_t i = 0; i < layer_num; ++i) {
    auto layer_successors
      = logical_dag_->GetSucceedingOpNodeNames(layer_set[i]);
    for (auto& layer_name : layer_successors) {
      successors.push_back(layer_name);
    }
  }

  // Find a succeeding group satisfying: (1) whose group_info is properly
  // initialized; (2) with a minimum level value in LogicalDag;
  PlacementGroupInfo group_info_from;
  int32_t minimum_level = std::numeric_limits<int32_t>::max();
  int32_t group_of_minimum_level = -1;
  for (auto& successor : successors) {
    auto group_name = strategy_descriptor_->group_from_layer(successor);
    auto group_id = strategy_descriptor_->group_id_from_name(group_name);
    auto successor_layer_set = strategy_descriptor_->layer_set(group_id);
    auto successor_layer_num = strategy_descriptor_->layer_num(group_id);
    int32_t minimum_layer_level = std::numeric_limits<int32_t>::max();
    for (int32_t layer_id = 0; layer_id < successor_layer_num; ++layer_id) {
      auto layer_name = successor_layer_set[layer_id];
      auto layer_level
        = logical_dag_->GetOpNodeLevel(layer_name);
      if (layer_level < minimum_layer_level) {
        minimum_layer_level = layer_level;
      }
    }
    if (strategy_descriptor_->group_info_is_initialized(group_id)) {
      if (minimum_layer_level < minimum_level) {
        group_of_minimum_level = group_id;
        minimum_level = minimum_layer_level;
      }
    }
  }
  CHECK(group_of_minimum_level != -1)
    << "No group can be found to be used for inheriting PlacementInfo";

  // Set the |gid| group to be kNaiveParallelOnSingleMachine or
  // kDataParallelOnMultipleMachines
  auto successor_machine_set
    = strategy_descriptor_->machine_set(group_of_minimum_level);
  int32_t begin = successor_machine_set.front();
  int32_t end = successor_machine_set.back();
  ParallelPolicy policy
    = end == begin ? kNaiveParallelOnSingleMachine
    : kDataParallelOnMultipleMachines;
  strategy_descriptor_->update_placement_info_with_machine_group(gid,
    begin, end, policy);
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::CompleteGroupInfoBackwardDependency(
  int32_t gid) {
  // Get all the layers preceding the layers in group |gid|
  auto layer_set = strategy_descriptor_->layer_set(gid);
  auto layer_num = strategy_descriptor_->layer_num(gid);
  CHECK(layer_num > 0);
  std::vector<std::string> predecessors;
  for (int32_t i = 0; i < layer_num; ++i) {
    auto layer_predecessors
      = logical_dag_->GetPrecedingOpNodeNames(layer_set[i]);
    for (auto& layer_name : layer_predecessors) {
      predecessors.push_back(layer_name);
    }
  }
  // Find a preceding group satisfying: (1) whose group_info is properly
  // initialized; (2) with a maximum level value in LogicalDag;
  PlacementGroupInfo group_info_from;
  int32_t maximum_level = -1;
  int32_t group_of_maximum_level = -1;
  for (auto& predecessor : predecessors) {
    auto group_name = strategy_descriptor_->group_from_layer(predecessor);
    auto group_id = strategy_descriptor_->group_id_from_name(group_name);
    auto predecessor_layer_set = strategy_descriptor_->layer_set(group_id);
    auto predecessor_layer_num = strategy_descriptor_->layer_num(group_id);
    CHECK(predecessor_layer_num > 0);
    int32_t maximum_layer_level = -1;
    for (int32_t layer_id = 0; layer_id < predecessor_layer_num; ++layer_id) {
      auto layer_name = predecessor_layer_set[layer_id];
      auto layer_level
        = logical_dag_->GetOpNodeLevel(layer_name);
      if (layer_level > maximum_layer_level) {
        maximum_layer_level = layer_level;
      }
    }
    if (strategy_descriptor_->group_info_is_initialized(group_id)) {
      if (maximum_layer_level > maximum_level) {
        group_of_maximum_level = group_id;
        maximum_level = maximum_layer_level;
      }
    }
  }
  CHECK(group_of_maximum_level != -1)
    << "No group can be found to be used for inheriting PlacementInfo";

  // Set the 'loss' related group as kDataParallelOnMultipleDevices or
  // kNaiveParallelOnMultipleDevices.
  // This group may include 'softmax' or 'loss' layer, for either case, we need
  // to force them to be DataParallel if they are on multiple devices.
  // TODO(jiyuan): however, if the group containing 'softmax' or 'loss' is
  // already initialized, we need also to check whether it confines to this
  // requirement.
  auto predecessor_device_set
    = strategy_descriptor_->device_set(group_of_maximum_level);
  int32_t begin = predecessor_device_set.front();
  int32_t end = predecessor_device_set.back();
  ParallelPolicy policy
    = (end == begin ? kNaiveParallelOnSingleDevice
    : kDataParallelOnMultipleDevices);
  strategy_descriptor_->update_placement_info_with_device_group(
    gid, begin, end, policy);
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::CompleteGroupInfo() {
  if (path_type_ != PathType::kDataPath) return;
  int32_t group_num = strategy_descriptor_->group_num();
  for (int32_t gid = 0; gid < group_num; ++gid) {
    if (!strategy_descriptor_->group_info_is_initialized(gid)) {
      // We ensure that the PlacementGroupInfo is complete in all the paths
      // except the kDataPath, in which the PlacementInfos of some particular
      // group are not specified, we need to complete them according to the
      // context.
      CHECK(path_type_ == PathType::kDataPath);
      if (gid == 0) {
        CompleteGroupInfoForwardDependency(gid);
      } else {
        CompleteGroupInfoBackwardDependency(gid);
      }
    }
  }
}
//template <typename Dtype>
//void PlacementGroupDag<Dtype>::VerifyGroupInModelParallel() {
//  int32_t group_num = strategy_descriptor_->group_num();
//  for (int32_t gid = 0; gid < group_num; ++gid) {
//    if (strategy_descriptor_->parallel_policy(gid)
//        == kModelParallelOnMultipleDevices) {
//      auto layer_num = strategy_descriptor_->layer_num(gid);
//      CHECK_EQ(layer_num, 1)
//        << "Group in kModelParallelOnMultipleDevices must have exactly 1 layer";
//    }
//  }
//}
template <typename Dtype>
void PlacementGroupDag<Dtype>::VerifyElemWiseLayerDependency() {
  DagIterator<LogicalDag<Dtype>, true> dag_iterator(*logical_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto layer_node
      = dynamic_cast<const OpNode<LayerMeta<Dtype>>*>(current_node);
    auto layer_meta = layer_node->op();
    auto layer_name = layer_node->node_name();
    auto& layer_placement_info = layer_meta->placement_info();
    if (layer_meta->layer()->IsElemWise()) {
      auto preceeding_layer_names
        = logical_dag_->GetPrecedingOpNodeNames(layer_name);
      CHECK(preceeding_layer_names.size() == 1)
        << "Element-wise layer can only have one predecessor: " << layer_name;
      auto preceeding_layer = logical_dag_->GetOpNode(preceeding_layer_names[0]);
      auto preceeding_layer_meta = preceeding_layer->op();
      auto& preceeding_placement_info = preceeding_layer_meta->placement_info();
      CHECK(layer_placement_info.EqualTo(preceeding_placement_info))
        << "Element-wise layer must have the same PlacementInfo with its "
        << "preceding layer: " << layer_name;
    }
  }
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::Build() {
  AddOpNodes();
  AddAndConnectDataNodes();
  AddStartAndEndNodes();
  PostProcessing();
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::SetPlacementInfoForLayer() {
  int32_t group_num = strategy_descriptor_->group_num();
  for (int32_t gid = 0; gid < group_num; ++gid) {
    auto name = strategy_descriptor_->name(gid);
    auto& placement_info = strategy_descriptor_->placement_info(gid);
    auto layer_num = strategy_descriptor_->layer_num(gid);
    auto layer_set = strategy_descriptor_->layer_set(gid);
    for (int32_t layer_id = 0; layer_id < layer_num; ++layer_id) {
      auto layer_name = layer_set[layer_id];
      auto layer_node = logical_dag_->GetOpNode(layer_name);
      auto layer_meta = layer_node->mutable_op();
      layer_meta->mutable_placement_info() = placement_info;
    }
  }
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::AddOpNodes() {
  int32_t group_num = strategy_descriptor_->group_num();
  for (int32_t gid = 0; gid < group_num; ++gid) {
    auto name = strategy_descriptor_->name(gid);
    auto& placement_info = strategy_descriptor_->placement_info(gid);
    auto op_node = AddOpNode(name, placement_info);
  }
}
template <typename Dtype>
OpNode<PlacementGroupMeta>* PlacementGroupDag<Dtype>::AddOpNode(
  const std::string& op_name,
  const PlacementInfo& placement_info) {
  auto op_node = NewOpNode(op_name);
  auto& group_meta = op_node->mutable_op();
  group_meta = std::make_shared<PlacementGroupMeta>();
  group_meta->mutable_placement_info() = placement_info;
  auto it = op_name_to_node_.find(op_name);
  CHECK(it == op_name_to_node_.end()) << "Duplicate op_name: " << op_name;
  op_name_to_node_.insert({ op_name, op_node});
  return op_node;
}
template <typename Dtype>
DataNode<BlobMeta>* PlacementGroupDag<Dtype>::AddDataNode(
    const std::string& blob_name) {
  auto data_node = NewDataNode(blob_name);
  auto& blob_meta = data_node->mutable_data();
  blob_meta = std::make_shared<BlobMeta>();
  blob_meta->mutable_name() = blob_name;
  auto it = data_name_to_node_.find(blob_name);
  CHECK(it == data_name_to_node_.end())
    << "Duplicate data_name: " << blob_name;
  data_name_to_node_.insert({ blob_name, data_node });
  return data_node;
}
template <typename Dtype>
void PlacementGroupDag<Dtype>::AddAndConnectDataNodes() {
  // For each layer in each group, finding its preceding layers. If the
  // the preceding layer is in another group, find the blob in-between. Create
  // a data node for the blob if it has no corresponding data node in the DAG.
  // Connect the edges if necessary.
  int32_t group_num = strategy_descriptor_->group_num();
  for (int32_t gid = 0; gid < group_num; ++gid) {
    auto group_name = strategy_descriptor_->name(gid);
    auto group_node = GetOpNode(group_name);
    auto layer_set = strategy_descriptor_->layer_set(gid);
    auto layer_num = layer_set.size();
    CHECK(layer_num != 0); 
    for (int32_t layer_id = 0; layer_id < layer_num; ++layer_id) {
      auto layer_name = layer_set[layer_id];
      auto predecessors
        = logical_dag_->GetPrecedingOpNodeNames(layer_name);
      for (auto& predecessor : predecessors) {
        auto predecessor_group
          = strategy_descriptor_->group_from_layer(predecessor);
        if (predecessor_group != group_name) {
          auto blob_names
            = logical_dag_->FindDataNodesInBetween(predecessor, layer_name);
          CHECK(blob_names.size() == 1);
          if (data_name_to_node_.count(blob_names[0]) > 0) {
            // The data_node of blob_names[0] already exists
            group_node->AddParent(data_name_to_node_[blob_names[0]]);
          } else {
            // Create the data_node of blob_names[0]
            auto data_node = AddDataNode(blob_names[0]);
            auto predecessor_group_node = GetOpNode(predecessor_group);
            data_node->AddParent(predecessor_group_node);
            group_node->AddParent(data_node);
          }
        }
      }
    }
  }
  // Add the data nodes of the blobs which directly connect to END node
  auto blob_names_without_successor
    = logical_dag_->GetPreceedingDataNodeNamesOfEndNode();
  for (auto& blob_name : blob_names_without_successor) {
    auto data_node = AddDataNode(blob_name);
    auto data_node_predecessors
      = logical_dag_->GetPreceedingOpNodeNamesOfDataNode(blob_name);
    CHECK(data_node_predecessors.size() == 1);
    auto preceeding_group_name
      = strategy_descriptor_->group_from_layer(data_node_predecessors[0]);
    auto preceeding_group_node = GetOpNode(preceeding_group_name);
    data_node->AddParent(preceeding_group_node);
  }
}
template <typename Dtype>
std::vector<std::string> PlacementGroupDag<Dtype>::GetLayerNamesInGroup(
  const std::string& group_name) const {
  auto group_id = strategy_descriptor_->group_id_from_name(group_name);
  auto layer_set = strategy_descriptor_->layer_set(group_id);
  return layer_set;
}
INSTANTIATE_CLASS(PlacementGroupDag);
}  // namespace oneflow
