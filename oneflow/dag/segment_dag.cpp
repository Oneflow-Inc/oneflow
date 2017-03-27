#include "dag/segment_dag.h"
#include <string>
#include <vector>
#include "common/common.h"
#include "common/stl_util.h"
#include "dag/node_meta.h"
#include "dag/dag_iterator.h"
#include "dag/logical_dag.h"
#include "dag/clustering_dag.h"
#include "layers/base_layer.h"
#include "common/string_pair.h"

namespace oneflow {
template <typename Dtype>
SegmentDag<Dtype>::SegmentDag(
  std::shared_ptr<LogicalDag<Dtype>> logical_dag, PathType path_type,
  const std::string& name) :
  Dag<EnvelopeMeta, SegmentMeta>(path_type, name), logical_dag_(logical_dag) {}

template <typename Dtype>
void SegmentDag<Dtype>::Build() {
  // Two situations:
  // (1) Merge model-parallel ops, the element-wise op can be merged with the
  //     previous op if they are both model-parallel;
  // (2) Merge data-parallel ops, data-parallel ops can be merged when they will
  //     not lag each other;
  // Besides, ops can be merged must have same parallel policy and same
  // device set(i.e., same PlacementGroupInfo). 

  // Construct a naive cluster structure, in which each cluster has a single
  // op_node in LogicalDag
  std::vector<std::vector<std::string>> naive_clusters;
  std::unordered_set<std::string> inherited_unchanged_clusters;
  DagIterator<LogicalDag<Dtype>, true> dag_iterator(*logical_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto node_name = current_node->node_name();
    naive_clusters.push_back({node_name});
    // Set Data Provider Node as unchanged, since we don't want to merge the
    // data node
    if (logical_dag_->IsFirstOpNode(current_node)) {
      inherited_unchanged_clusters.insert(node_name);
    }
  }

  // (1) Let's try to merge the model-parallel ops
  ClusteringDag<Dtype> merged_model_parallel_dag(
    *logical_dag_,
    naive_clusters,
    inherited_unchanged_clusters,
    MergeWayInClusteringDag::kMergeModelParallel);
  merged_model_parallel_dag.Build();
  merged_model_parallel_dag.PrintDag("merged_model_parallel_dag");

  // (2) Let's try to merge data-parallel ops
  auto clusters_after_merging_model_parallel
    = merged_model_parallel_dag.GetResultedClusters();

  ClusteringDag<Dtype> merged_data_parallel_dag(
    *logical_dag_,
    clusters_after_merging_model_parallel,
    inherited_unchanged_clusters,
    MergeWayInClusteringDag::kMergeDataParallel);
  merged_data_parallel_dag.Build();
  merged_data_parallel_dag.PrintDag(
    "merged_data_parallel_dag");

  // CloneOpNodes
  DagIterator<ClusteringDag<Dtype>, true>
    dag_iter(merged_data_parallel_dag);
  for (dag_iter.First(); !dag_iter.IsDone(); dag_iter.Next()) {
    auto current_node = dag_iter.CurrentNode();
    auto current_name = current_node->node_name();
    // Skip the non-kOpNode groups
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto clustering_node
      = dynamic_cast<const OpNode<ClusteringMeta>*>(current_node);
    CHECK_NOTNULL(clustering_node);
    auto clustering_meta = clustering_node->op();
    auto& layer_names = clustering_meta->layer_names();
    CHECK(layer_names.size() > 0);
    auto layer_node = logical_dag_->GetOpNode(layer_names[0]);
    auto layer_meta = layer_node->op();
    auto& placement_info = layer_meta->placement_info();
    auto segment_node = AddOpNode(layer_names, placement_info);
  }

  CloneDataNodes();
  AddStartAndEndNodes();
  PostProcessing();

  CollectComputeSegments();

  PrintDag(name_);

  if (path_type_ == PathType::kDataPath) {
    // NOTE(jiyuan): only check if it is in kDataPath, since in other paths, the
    // strategy is not provided by users.
    VerifyTopology();
  }
}

template <typename Dtype>
void SegmentDag<Dtype>::VerifyTopology() {
  DagIterator<SegmentDag<Dtype>, true> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto segment_node_ptr
      = dynamic_cast<const OpNode<SegmentMeta>*>(current_node);
    CHECK_NOTNULL(segment_node_ptr);
    auto segment_meta = segment_node_ptr->op();
    auto segment_name = segment_node_ptr->node_name();
    auto& segment_placement_info = segment_meta->placement_info();

    auto segment_successors
      = GetSucceedingOpNodeNames(segment_name);
    for (auto& next_segment_name : segment_successors) {
      auto next_segment_node = GetOpNode(next_segment_name);
      if (IsEnd(next_segment_node)) continue;  // Skip the end node
      auto next_segment_meta = next_segment_node->op();
      auto& next_placement_info = next_segment_meta->placement_info();

      TheSameOrNoOverlapMachine(
        segment_placement_info,
        segment_name,
        next_placement_info,
        next_segment_name);

      NoConsecutiveDataParallelWithSameMachineSet(
        segment_placement_info,
        segment_name,
        next_placement_info,
        next_segment_name,
        segment_successors);

      ConsecutiveDataParallelShardingCondition(
        segment_placement_info,
        segment_name,
        next_placement_info,
        next_segment_name);

      NoModelParallelBeforeDataParallel(
        segment_placement_info,
        segment_name,
        next_placement_info,
        next_segment_name);
    }
  }
  DataSegmentAndLossSegmentShardingCondition();
  SplittingBlobWithinSegmentCondition();
  ConcatLayerWithinSegmentCondition();
}

template <typename Dtype>
void SegmentDag<Dtype>::CollectComputeSegments() {
  DagIterator<SegmentDag<Dtype>, true> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto segment_node_ptr
      = dynamic_cast<const OpNode<SegmentMeta>*>(current_node);
    CHECK_NOTNULL(segment_node_ptr);
    auto segment_meta = segment_node_ptr->op();
    auto segment_name = segment_node_ptr->node_name();
    auto& placement_info = segment_meta->placement_info();
    auto& device_set = placement_info.device_set();
    if (device_set.size() > 0) {
      compute_segments_.push_back(segment_name);
    }
  }
}

template <typename Dtype>
std::vector<std::string> SegmentDag<Dtype>::GetComputeSegments() const {
  return compute_segments_;
}

template <typename Dtype>
void SegmentDag<Dtype>::SplittingBlobWithinSegmentCondition() const {
  // TODO(jiyuan): for data provider segment, allow cross-segment blob sharing
  DagIterator<SegmentDag<Dtype>, true> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto segment_node_ptr
      = dynamic_cast<const OpNode<SegmentMeta>*>(current_node);
    CHECK_NOTNULL(segment_node_ptr);
    auto segment_meta = segment_node_ptr->op();
    auto segment_name = segment_node_ptr->node_name();
    auto envelope_names = GetSucceedingDataNodeNames(segment_name);
    // If there are multiple succeeding envelopes for the |segment_name|, ensure
    // the envelopes do not overlap with each other. This is equivalent to the
    // requirement of no cross-segment blob splitting.
    std::unordered_set<std::string> blob_set;
    if (envelope_names.size() > 1) {
      for (auto envelope_name : envelope_names) {
        auto envelope_node = GetDataNode(envelope_name);
        auto blob_names = envelope_node->data()->blob_names();
        for (auto blob_name : blob_names) {
          CHECK(blob_set.count(blob_name) == 0)
            << "SplittingBlobWithinSegmentCondition fail";
          blob_set.insert(blob_name);
        }
      }
    }
  }
}

template <typename Dtype>
void SegmentDag<Dtype>::ConcatLayerWithinSegmentCondition() const {
  DagIterator<LogicalDag<Dtype>, true> dag_iterator(*logical_dag_);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto node_name = current_node->node_name();
    auto layer_node
      = dynamic_cast<const OpNode<LayerMeta<Dtype>>*>(current_node);
    CHECK_NOTNULL(layer_node);
    // Ensure the ConcatLayer and its predecessors are merged into the same
    // segment.
    if (layer_node->op()->type() == "Concat") {
      auto segment_it = layer_name_to_segment_name_.find(node_name);
      CHECK(segment_it != layer_name_to_segment_name_.end());
      auto segment_name = segment_it->second;
      auto preceding_layer_names
        = logical_dag_->GetPrecedingOpNodeNames(node_name);
      for (auto preceding_layer_name : preceding_layer_names) {
        auto preceding_segment_it
          = layer_name_to_segment_name_.find(preceding_layer_name);
        CHECK(preceding_segment_it != layer_name_to_segment_name_.end());
        auto preceding_segment_name = preceding_segment_it->second;
        CHECK(segment_name == preceding_segment_name)
          << "ConcatLayerWithinSegmentCondition fails";
      }
    }
  }
}

template <typename Dtype>
void SegmentDag<Dtype>::TheSameOrNoOverlapMachine(
  const PlacementInfo& first_placement_info,
  const std::string& first_segment_name,
  const PlacementInfo& second_placement_info,
  const std::string& second_segment_name) const {
  bool is_same = stl::VectorEqual(
    first_placement_info.machine_set(), second_placement_info.machine_set());
  bool no_overlap = stl::VectorNoOverlap(
    first_placement_info.machine_set(), second_placement_info.machine_set());
  if (!(is_same || no_overlap)) {
    LOG(FATAL)
      << "Two consecutive segments must: either (1) they"
      << " have the same machine set; or (2) totally no overlapping."
      << " Please check the PlacementGroups: "
      << first_segment_name << " and " << second_segment_name;
  }
}

template <typename Dtype>
void SegmentDag<Dtype>::NoConsecutiveDataParallelWithSameMachineSet(
  const PlacementInfo& first_placement_info,
  const std::string& first_segment_name,
  const PlacementInfo& second_placement_info,
  const std::string& second_segment_name,
  const std::vector<std::string>& all_successors_of_first_segment) const {
  if (first_placement_info.EqualTo(second_placement_info)
    && first_placement_info.parallel_policy() == kDataParallelOnMultipleDevices) {
    // Invalid except 3 cases:
    // (1) the first segment is data provider node; OR
    // (2) the second segment is loss node; OR
    // (3) the first has other children which can not be merged
    //     with second_segment_name

    // (1)
    bool first_is_data_provider
      = IsFirstOpNode(first_segment_name);
    if (first_is_data_provider) return;

    // (2)
    bool second_is_loss_node
      = IsLastOpNode(second_segment_name);
    if (second_is_loss_node) return;

    // (3)
    bool has_other_siblings_not_mergable = false;
    for (auto& segment_successor : all_successors_of_first_segment) {
      if (segment_successor == second_segment_name) continue;
      auto segment_successor_node = GetOpNode(segment_successor);
      auto segment_successor_meta = segment_successor_node->op();
      auto& segment_successor_placement_info
        = segment_successor_meta->placement_info();
      if (!first_placement_info.EqualTo(segment_successor_placement_info)) {
        has_other_siblings_not_mergable = true;
      }
    }
    if (has_other_siblings_not_mergable) return;
    LOG(FATAL) << "Two consecutive data-parallel segments with the"
      << " same PlacementInfo are not allowed, except (1) the first is"
      << " a data segment, or (2) the second is a loss segment, or "
      << " (3) the second has other siblings not mergable. Otherwise, should"
      << " have been merged in segmentDag: " << first_segment_name
      << " and " << second_segment_name;
  }
}

template <typename Dtype>
void SegmentDag<Dtype>::ConsecutiveDataParallelShardingCondition(
  const PlacementInfo& first_placement_info,
  const std::string& first_segment_name,
  const PlacementInfo& second_placement_info,
  const std::string& second_segment_name) const {
  bool no_overlap = stl::VectorNoOverlap(
    first_placement_info.machine_set(), second_placement_info.machine_set());
  auto first_parallel_policy = first_placement_info.parallel_policy();
  auto second_parallel_policy = second_placement_info.parallel_policy();
  bool both_are_data_parallel
    = (first_parallel_policy == second_parallel_policy)
    && (first_parallel_policy == kDataParallelOnMultipleDevices);

  if (no_overlap && both_are_data_parallel) {
    bool machine_size_no_equal
      = (first_placement_info.machine_set().size()
      != second_placement_info.machine_set().size());
    if (machine_size_no_equal) {
      bool first_is_data_provider
        = IsFirstOpNode(first_segment_name);
      bool second_is_loss_node
        = IsFirstOpNode(second_segment_name);
      if (!(first_is_data_provider && second_is_loss_node)) {
        // Specific to the relay mode
        LOG(FATAL) << "Two consecutive data-parallel segments with"
          << " different number of machines in the machine set are not"
          << " allowed. Please check the segments: " << first_segment_name
          << " and " << second_segment_name;
      }
    }
  }
}

template <typename Dtype>
void SegmentDag<Dtype>::NoModelParallelBeforeDataParallel(
  const PlacementInfo& first_placement_info,
  const std::string& first_segment_name,
  const PlacementInfo& second_placement_info,
  const std::string& second_segment_name) const {
  auto first_parallel_policy = first_placement_info.parallel_policy();
  auto second_parallel_policy = second_placement_info.parallel_policy();
  if (first_parallel_policy == kModelParallelOnMultipleDevices
    && second_parallel_policy == kDataParallelOnMultipleDevices) {
    bool second_is_loss_node = IsLastOpNode(second_segment_name);
    if (!second_is_loss_node) {
      LOG(FATAL) << "Segment with kModelParallel can not occur before"
        << " the segment with kDataParallel, except the last loss segment;"
        << " If you indeed have this requirement, please contact the authors"
        << " of this software. We would like to know more.";
    }
  }
}

template <typename Dtype>
void SegmentDag<Dtype>::DataSegmentAndLossSegmentShardingCondition() const {
  auto data_segment_names = GetFirstOpNames();
  auto loss_segment_names = GetLastOpNames();
  //// NOTE(Chonglin): If this is PSDag, there may be multiple gradient provider
  //CHECK_EQ(IsPSDag() || data_segment_names.size(), 1)
  //  << "Currently, assuming a single data provider name";
  CHECK_EQ(data_segment_names.size(), 1);
  auto data_segment_node = GetOpNode(data_segment_names[0]);
  auto data_segment_meta = data_segment_node->op();
  auto& data_placement_info = data_segment_meta->placement_info();
  auto data_device_set = data_placement_info.device_set();
  auto data_machine_set = data_placement_info.machine_set();
  int32_t data_machine_size = data_machine_set.size();

  for (auto& loss_segment_name : loss_segment_names) {
    auto loss_segment_node = GetOpNode(loss_segment_name);
    auto loss_segment_meta = loss_segment_node->op();
    auto& loss_placement_info = loss_segment_meta->placement_info();
    auto loss_device_set = loss_placement_info.device_set();
    auto loss_machine_set = loss_placement_info.machine_set();
    int32_t loss_machine_size = loss_machine_set.size();

    CHECK(data_machine_size == loss_machine_size)
      << "We require that the numbers of machines for data segment and for loss "
      << "segment are equal";

    //CHECK(data_machine_size >= loss_machine_size)
    //  << "The number of machines in loss group is not allowed to exceed that of"
    //  << " data group";
    //CHECK_EQ(data_machine_size % loss_machine_size, 0)
    //  << "The number of machines in data group should be able to be divided by"
    //  << " that of loss group";
  }
}

template <typename Dtype>
void SegmentDag<Dtype>::CloneDataNodes() {
  // Collect the intermediate blobs between any pair of neighboring segments
  using SegmentSegmentPair = StringPair;
  std::unordered_map<SegmentSegmentPair, std::vector<std::string>>
    segment_to_segment_blobs;
  std::unordered_map<SegmentSegmentPair, std::unordered_set<std::string>>
    segment_to_segment_blob_set;
  std::unordered_map<std::string, std::vector<std::string>>
    segment_to_blobs_no_successor;
  for (auto& name_node_pair : op_name_to_node_) {
    auto segment_name = name_node_pair.first;
    auto segment_node = name_node_pair.second;
    auto segment_meta = segment_node->op();
    auto& layer_names = segment_meta->layer_names();
    for (auto& layer_name : layer_names) {
      auto successors
        = logical_dag_->GetSucceedingOpNodeNames(layer_name);
      if (successors.size() == 0) {
        // This layer has no succeeding layer
        auto blobs_no_successor
          = logical_dag_->GetSucceedingDataNodeNames(layer_name);
        //CHECK(blobs_no_successor.size(), 1);
        auto segment_to_blobs_it
          = segment_to_blobs_no_successor.find(segment_name);
        if (segment_to_blobs_it == segment_to_blobs_no_successor.end()) {
          segment_to_blobs_no_successor.insert({
            segment_name, blobs_no_successor });
        } else {
          segment_to_blobs_it->second.push_back(blobs_no_successor[0]);
        }
      } else {
        // This layer has succeeding layer
        for (auto& successor : successors) {
          //CHECK(layer_name_to_segment_name_.count(layer_name) > 0);
          auto successor_segment
            = layer_name_to_segment_name_[successor];
          if (successor_segment != segment_name) {
            auto blob_names
              = logical_dag_->FindDataNodesInBetween(layer_name, successor);
            CHECK(blob_names.size() == 1)
              << "In LogicalDag, every two neighboring layers have exactly one"
              << "intermediate data blob";
            SegmentSegmentPair segment_to_segment{
              segment_name, successor_segment };
            for (auto blob_name : blob_names) {
              if (blob_to_segment_pair_.count(blob_name) == 0) {
                blob_to_segment_pair_.insert({ blob_name, segment_to_segment });
              }
            }
            auto segment_to_segment_it
              = segment_to_segment_blob_set.find(segment_to_segment);
            if (segment_to_segment_it == segment_to_segment_blob_set.end()) {
              // This segment-segment pair occurs for the first time
              segment_to_segment_blobs.insert({ segment_to_segment, blob_names });
              segment_to_segment_blob_set.insert({
                segment_to_segment, { blob_names[0] } });
            } else {
              // This segment-segment pair occurs previously, check whether the
              // intermediate blob is a fresh one
              if (segment_to_segment_it->second.count(blob_names[0]) == 0) {
                segment_to_segment_it->second.insert(blob_names[0]);
                segment_to_segment_blobs[segment_to_segment].push_back(
                  blob_names[0]);
              }
            }
          }
        }
      }
    }
  }
  // Add data node between neighboring segments if necessary
  for (auto& segment_to_segment_blobs_pair : segment_to_segment_blobs) {
    auto segment = segment_to_segment_blobs_pair.first.first;
    auto succeeding_segment = segment_to_segment_blobs_pair.first.second;
    auto blob_names = segment_to_segment_blobs_pair.second;
    auto envelope_name = build_envelope_name(blob_names);

    auto segment_node = GetOpNode(segment);
    auto succeeding_segment_node = GetOpNode(succeeding_segment);

    if (data_name_to_node_.count(envelope_name) > 0) {
      // The data_node of envelope_name already exists
      succeeding_segment_node->AddParent(data_name_to_node_[envelope_name]);
    } else {
      // Create the data_node of envelope_name
      auto data_node = AddDataNode(blob_names);
      data_node->AddParent(segment_node);
      succeeding_segment_node->AddParent(data_node);
    }
  }
  // Add data node without succeeding segments (i.e., directly connecting to end)
  for (auto segment_to_blobs_pair : segment_to_blobs_no_successor) {
    auto segment = segment_to_blobs_pair.first;
    auto blob_names = segment_to_blobs_pair.second;
    auto segment_node = GetOpNode(segment);
    auto data_node = AddDataNode(blob_names);
    data_node->AddParent(segment_node);
  }
}

template <typename Dtype>
SegmentSegmentPair SegmentDag<Dtype>::GetSegmentPairAroundBlob(
  const std::string& blob_name) const {
  auto segment_pair_it = blob_to_segment_pair_.find(blob_name);
  CHECK(segment_pair_it != blob_to_segment_pair_.end());
  return segment_pair_it->second;
}

template <typename Dtype>
std::vector<std::string> SegmentDag<Dtype>::GetInputBlobs(
  const std::string& segment) const {
  auto envelope_names = GetPrecedingDataNodeNames(segment);
  std::vector<std::string> blobs;
  for (auto& envelope_name : envelope_names) {
    auto data_node = GetDataNode(envelope_name);
    auto blob_names = data_node->data()->blob_names();
    blobs.insert(blobs.end(), blob_names.begin(), blob_names.end());
  }
  return blobs;
}

template <typename Dtype>
std::vector<std::string> SegmentDag<Dtype>::GetOutputBlobs(
  const std::string& segment) const {
  auto envelope_names = GetSucceedingDataNodeNames(segment);
  std::vector<std::string> blobs;
  for (auto& envelope_name : envelope_names) {
    auto data_node = GetDataNode(envelope_name);
    auto blob_names = data_node->data()->blob_names();
    blobs.insert(blobs.end(), blob_names.begin(), blob_names.end());
  }
  return blobs;
}

template <typename Dtype>
OpNode<SegmentMeta>* SegmentDag<Dtype>::AddOpNode(
  const std::vector<std::string>& layer_names,
  const PlacementInfo& placement_info) {
  auto segment_name = build_segment_name(layer_names);
  for (auto& layer_name : layer_names) {
    if (layer_name_to_segment_name_.count(layer_name) > 0) {
      // the layer already exists
      layer_name_to_segment_name_[layer_name] = segment_name;
    } else {
      // a new layer
      layer_name_to_segment_name_.insert({ layer_name, segment_name });
    }
  }
  // NOTE(jiyuan): segments in non-kDataPath do not need BP; 
  // Currently, suppose kDataPath is for training purpose. In the future, the
  // inference-only job will set the has_BP to be false for all segments.
  // In kDataPath, a segment needs BP once one of its included layer needs BP.
  bool has_BP;
  if (path_type_ == PathType::kDataPath) {
    has_BP = false;
    for (auto& layer_name : layer_names) {
      auto layer_node = logical_dag_->GetOpNode(layer_name);
      auto layer_meta = layer_node->op();
      if (layer_meta->has_BP()) {
        has_BP = true;
      }
    }
  } else {
    // not kDataPath, no need BP
    has_BP = false;
  }

  auto op_node = NewOpNode(segment_name);
  auto& segment_meta = op_node->mutable_op();
  segment_meta = std::make_shared<SegmentMeta>();
  segment_meta->mutable_layer_names() = layer_names;
  segment_meta->mutable_placement_info() = placement_info;
  segment_meta->mutable_has_BP() = has_BP;
  auto it = op_name_to_node_.find(segment_name);
  CHECK(it == op_name_to_node_.end()) << "Duplicate op_name: " << segment_name;
  op_name_to_node_.insert({ segment_name, op_node });
  return op_node;
}

template <typename Dtype>
DataNode<EnvelopeMeta>* SegmentDag<Dtype>::AddDataNode(
  const std::vector<std::string>& blob_names) {
  std::string envelope_name = build_envelope_name(blob_names);

  auto data_node = NewDataNode(envelope_name);
  auto& envelope_meta = data_node->mutable_data();
  envelope_meta = std::make_shared<EnvelopeMeta>();
  envelope_meta->mutable_blob_names() = blob_names;
  auto it = data_name_to_node_.find(envelope_name);
  CHECK(it == data_name_to_node_.end())
    << "Duplicate data_name: " << envelope_name;
  data_name_to_node_.insert({ envelope_name, data_node });
  return data_node;
}

template <typename Dtype>
bool SegmentDag<Dtype>::NeedNullUpdate(const std::string& segment_name) const {
  bool has_temp_vars = false;
  bool has_model_vars = false;
  HasModelOrTempVars(segment_name, &has_model_vars, &has_temp_vars);
  return has_temp_vars && !has_model_vars;
}

template <typename Dtype>
bool SegmentDag<Dtype>::NeedModelUpdate(const std::string& segment_name) const {
  bool has_temp_vars = false;
  bool has_model_vars = false;
  HasModelOrTempVars(segment_name, &has_model_vars, &has_temp_vars);
  return has_model_vars;
}

template <typename Dtype>
void SegmentDag<Dtype>::HasModelOrTempVars(const std::string& segment_name,
  bool* has_model_vars, bool* has_temp_vars) const {
  auto segment_node = GetOpNode(segment_name);
  auto segment_meta = segment_node->op();
  auto layer_names = segment_meta->layer_names();
  *has_temp_vars = false;
  *has_model_vars = false;
  for (auto& layer_name : layer_names) {
    auto layer_node = logical_dag_->GetOpNode(layer_name);
    auto layer_meta = layer_node->op();
    auto layer = layer_meta->layer();
    *has_temp_vars = *has_temp_vars || layer->GetTempVars().size() > 0;
    *has_model_vars = *has_model_vars || layer->GetModelVars().size() > 0;
  }
  return;
}

template <typename Dtype>
std::vector<int32_t> SegmentDag<Dtype>::DeviceSetOfSegment(
  const std::string& segment_name) const {
  auto segment_node = GetOpNode(segment_name);
  auto segment_meta = segment_node->op();
  auto placement_info = segment_meta->placement_info();
  return placement_info.device_set();
}

template <typename Dtype>
ParallelPolicy SegmentDag<Dtype>::ParallelPolicyOfSegment(
  const std::string& segment_name) const {
  auto segment_node = GetOpNode(segment_name);
  auto segment_meta = segment_node->op();
  auto placement_info = segment_meta->placement_info();
  return placement_info.parallel_policy();
}

template <typename Dtype>
std::string SegmentDag<Dtype>::build_envelope_name(
  const std::vector<std::string>& blob_names) const {
  std::string envelope_name = "";
  for (auto blob_name : blob_names) {
    if (envelope_name == "") {
      envelope_name = blob_name;
    } else {
      envelope_name += "_" + blob_name;
    }
  }
  return envelope_name;
}

template <typename Dtype>
std::string SegmentDag<Dtype>::build_segment_name(
  const std::vector<std::string>& layer_names) const {
  std::string segment_name = "";
  for (auto& layer_name : layer_names) {
    if (segment_name == "") {
      segment_name = layer_name;
    } else {
      segment_name += "_" + layer_name;
    }
  }
  return segment_name;
}
INSTANTIATE_CLASS(SegmentDag);
}  // namespace oneflow
