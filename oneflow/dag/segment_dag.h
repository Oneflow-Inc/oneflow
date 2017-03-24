#ifndef _DAG_SEGMENT_DAG_H_
#define _DAG_SEGMENT_DAG_H_
#include <vector>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "dag/dag_node.h"
#include "dag/dag.h"
#include "common/string_pair.h"
#include "context/placement_info.h"

namespace oneflow {
class EnvelopeMeta;

class SegmentMeta;

/*
template <typename DAG, bool isconst = false>
class DagIterator;

template <typename DAG, bool isconst = false>
class DagReverseIterator;
*/

template <typename Dtype>
class LogicalDag;

using SegmentSegmentPair = StringPair;

template <typename Dtype>
class SegmentDag : public Dag<EnvelopeMeta, SegmentMeta> {
  friend class DagIterator<SegmentDag<Dtype>>;
  friend class DagIterator<SegmentDag<Dtype>, true>;
  friend class DagReverseIterator<SegmentDag<Dtype>>;
  friend class DagReverseIterator<SegmentDag<Dtype>, true>;
public:
  SegmentDag(std::shared_ptr<LogicalDag<Dtype>> logical_dag, PathType path_type,
    const std::string& name = "segment_dag");
  ~SegmentDag() {}

  void Build();
  SegmentSegmentPair GetSegmentPairAroundBlob(const std::string& blob_name) const;
  std::vector<std::string> GetInputBlobs(const std::string& segment_name) const;
  std::vector<std::string> GetOutputBlobs(const std::string& segment_name) const;

  // Whether the |segment_name| just needs a NullUpate module: just has kTemp
  // variables, but without kModel variables.
  bool NeedNullUpdate(const std::string& segment_name) const;
  // Whether the |segment_name| needs a ModelUpdate module: has kModel variables
  bool NeedModelUpdate(const std::string& segment_name) const;

  // Get the device_set assigned to |segment_name|, it is possible to return an
  // empty vector.
  std::vector<int32_t> DeviceSetOfSegment(const std::string& segment_name) const;
  ParallelPolicy ParallelPolicyOfSegment(const std::string& segment_name) const;

  std::vector<std::string> GetComputeSegments() const;

private:
  std::shared_ptr<LogicalDag<Dtype>> logical_dag_;
  std::unordered_map<std::string, std::string> layer_name_to_segment_name_;
  std::unordered_map<std::string, SegmentSegmentPair> blob_to_segment_pair_;
  std::vector<std::string> compute_segments_;

  void CloneDataNodes();
  void VerifyTopology();
  void CollectComputeSegments();

  void TheSameOrNoOverlapMachine(
    const PlacementInfo& first_placement_info,
    const std::string& first_segment_name,
    const PlacementInfo& second_placement_info,
    const std::string& second_segment_name) const;

  void NoConsecutiveDataParallelWithSameMachineSet(
    const PlacementInfo& first_placement_info,
    const std::string& first_segment_name,
    const PlacementInfo& second_placement_info,
    const std::string& second_segment_name,
    const std::vector<std::string>& all_successors_of_first_segment) const;

  void ConsecutiveDataParallelShardingCondition(
    const PlacementInfo& first_placement_info,
    const std::string& first_segment_name,
    const PlacementInfo& second_placement_info,
    const std::string& second_segment_name) const;

  void NoModelParallelBeforeDataParallel(
    const PlacementInfo& first_placement_info,
    const std::string& first_segment_name,
    const PlacementInfo& second_placement_info,
    const std::string& second_segment_name) const;

  void DataSegmentAndLossSegmentShardingCondition() const;
  // When a blob needs to be split, ensure all its consumers are in the same 
  // segment. Currently, we don't allow cross-segment blob splitting.
  void SplittingBlobWithinSegmentCondition() const;
  // For ConcatLayer, ensure its producers are in the same segment. Currently, 
  // we don't allow cross-segment blob concat.
  void ConcatLayerWithinSegmentCondition() const;

  ONode* AddOpNode(const std::vector<std::string>& layer_names,
    const PlacementInfo& placement_info);

  DNode* AddDataNode(const std::vector<std::string>& blob_names);

  void HasModelOrTempVars(const std::string& segment_name, bool* has_model_vars,
    bool* has_temp_vars) const;

  std::string build_envelope_name(
    const std::vector<std::string>& blob_names) const;
  std::string build_segment_name(
    const std::vector<std::string>& layer_names) const;

  SegmentDag(const SegmentDag& other) = delete;
  SegmentDag& operator=(const SegmentDag& other) = delete;
};
}  // namespace oneflow
#endif  // _DAG_SEGMENT_DAG_H_
