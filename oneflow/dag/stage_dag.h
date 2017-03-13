#ifndef _DAG_STAGE_DAG_H_
#define _DAG_STAGE_DAG_H_
#include <vector>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include "dag/dag_node.h"
#include "dag/dag.h"
#include "common/string_pair.h"
/*
StageDag expands SegmentDag, it describes a machine-level connection. 
If a segment in a SegmentDag is distributed to multiple machines, we expand it 
into multiple stages. Each stage corresponds to exactly one segment. The major
challenge in building StageDag is how to properly connect the stages according
to the corresponding segment's placment_info. Concretely, suppose segment A and
B are consecutive in the SegmentDag, each of them is distributed to a set of 
machines. In other words, segment A and segment B both are expanded to a few
stages. We know that segment B should receive input from segment A. How do 
we connect the stages generated from segment B and the stages generated from
segment A? should it use a all-to-all connection or a one-to-one connection?

StageDag can be used to infer whether a segment needs net communication in 
forward propagation or in the back-propagation. A stage resides on exactly one
machine, however, multiple stages may share the same machine.
*/

namespace caffe {
class EnvelopeMeta;

class StageMeta;

template <typename DAG, bool isconst = false>
class DagIterator;

template <typename DAG, bool isconst = false>
class DagReverseIterator;

template <typename Dtype>
class LogicalDag;

template <typename Dtype>
class SegmentDag;

using StageSegmentPair = StringPair;
template <typename Dtype>
class StageDag : public Dag<EnvelopeMeta, StageMeta> {
friend class DagIterator<StageDag<Dtype>>;
friend class DagIterator<StageDag<Dtype>, true>;
friend class DagReverseIterator<StageDag<Dtype>>;
friend class DagReverseIterator<StageDag<Dtype>, true>;
public:
  StageDag(
    std::shared_ptr<LogicalDag<Dtype>> logical_dag,
    std::shared_ptr<SegmentDag<Dtype>> segment_dag, PathType path_type,
    const std::string& name = "stage_dag");
  ~StageDag() {}
  void Build();

  std::vector<std::string> GetStageNamesFromSegmentName(
    const std::string& segment_name) const;

  std::string GetSegmentNameFromStageName(const std::string& stage_name) const;

  int32_t machine_id(const std::string& stage_name) const;

private:
  // Managing the mapping between stage and segment
  class StageSegmentMap {
  public:
    StageSegmentMap() = default;
    ~StageSegmentMap() = default;
    void AddStageSegment(const std::string& stage, const std::string& segment);
    std::string SegmentFromStage(const std::string& stage) const;
    std::vector<std::string> StagesFromSegment(const std::string& segment) const;
  private:
    std::unordered_map<std::string, std::vector<std::string>> segment_to_stages_;
    std::unordered_map<std::string, std::string> stage_to_segment_;
  };

private:
  std::shared_ptr<LogicalDag<Dtype>> logical_dag_;
  std::shared_ptr<SegmentDag<Dtype>> segment_dag_;

  StageSegmentMap stage_segment_map_;

  std::unordered_map<StageSegmentPair, std::vector<StageSegmentPair>>
    stage_segment_to_successors_;
  std::unordered_map<std::string,
    std::unordered_map<std::string, std::string>> stage_to_stage_;

  void CreateStagesForSegment(const std::string& segment_name);
  void CreateStages();
  OpNode<StageMeta>* AddOpNode(const std::string& op_name,
    int32_t machine_id, const std::string& segment_name);

  void UpdateStageToStage(
    const std::string& stage_from,
    const std::string& stage_to,
    const std::string& envelope_name);

  void CollectStageToStageConnection();
  void OneToOneConnect(
    const std::string& first_segment_name,
    const std::vector<std::string>& first_stage_name,
    const std::string& second_segment_name,
    const std::vector<std::string>& second_stage_name,
    const std::string& envelope_name);

  //void ContractConnect(
  //  const std::string& first_segment_name,
  //  const std::vector<std::string>& first_stage_names,
  //  const std::string& second_segment_name,
  //  const std::vector<std::string>& second_stage_names,
  //  const std::string& envelope_name);

  void FullConnect(
    const std::string& first_segment_name,
    const std::vector<std::string>& first_stage_name,
    const std::string& second_segment_name,
    const std::vector<std::string>& second_stage_name,
    const std::string& envelope_name);

  void ConnectStageToStage();
  DataNode<EnvelopeMeta>* AddDataNode(const std::string& data_name,
    const std::vector<std::string>& blob_names);

  void AddStageSegmentPair(
    const std::string& stage_name,
    const std::string& segment_name,
    const std::string& next_stage_name,
    const std::string& next_segment_name);

  std::string build_stage_name(
    const std::string& segment_name,
    int32_t machine_id) const;
  std::string build_envelope_name(
    const std::string& segment_envelope_name,
    int32_t from_machine_id,
    int32_t to_machine_id) const;

  StageDag(const StageDag& other) = delete;
  StageDag& operator=(const StageDag& other) = delete;
};
}  // namespace caffe
#endif  // _DAG_STAGE_DAG_H_
