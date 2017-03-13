#ifndef _DAG_PIPE_DAG_H_
#define _DAG_PIPE_DAG_H_
#include <vector>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include "dag/dag_node.h"
#include "dag/dag.h"
#include "common/string_pair.h"
#include "dag/boxing_info.h"

/*
PipeDag expands StageDag, it describes the device-level connections. If a stage
contains multiple devices, it is expanded into several pipes, each of which
corresponds to a distinct device. We restrict the pipes in the same stage have
exactly the same communication behavior. Therefore, a straightforward expansion
from StageDag should be sufficient for connection purpose.

However, we need to properly insert pipes for host-device copy, network
communication, and pipes for boxing purpose.
Firstly,we insert copy task at proper positions for computing task. In most
cases, a computing actor needs two copy actors, one before the computing pipe
and the other after the computing pipe. However, there are other corner cases
needing to be carefully handled with.

Secondly, we insert net pipe in proper position.

Boxing pipe is added to break the potential deadlock while multiple tasks
concurrently request the limited device memory.
(see https://en.wikipedia.org/wiki/Deadlock for Coffman conditions of deadlock).

Suppose A and B are on the same machine, let's take a look at several examples.
(1) if A and B have a boxing actor in-between:
in_copy <-> computing_A <-> out_copy <-> boxing <-> in_copy <-> computing_B <-> out_copy
(2) if A and B have no boxing actor in-between, however they are on different devices:
in_copy <-> computing_A <-> out_copy <-> in_copy <-> computing_B <-> out_copy
(3) if A and B have no boxing actor in-between, they are on the same device:
in_copy <-> computing_A <-> (TODO) <-> computing_B <-> out_copy

The case (3) indicates the intermediate data are not necessary to be moved out
from the device. We only need a in-device memory copy. This needs to be supported
in the future.

In summary, the major challenge of building PipeDag is how to properly insert
net pipe and boxing pipe.
*/
namespace caffe {
enum class TaskType;

class EnvelopeMeta;

class PipeMeta;

template <typename DAG, bool isconst = false>
class DagIterator;

template <typename DAG, bool isconst = false>
class DagReverseIterator;

template <typename Dtype>
class SegmentDag;

template <typename Dtype>
class StageDag;

using SegmentSegmentPair = StringPair;
using StageStagePair = StringPair;

template <typename Dtype>
class PipeDag : public Dag<EnvelopeMeta, PipeMeta> {
  friend class DagIterator<PipeDag<Dtype>>;
  friend class DagIterator<PipeDag<Dtype>, true>;
  friend class DagReverseIterator<PipeDag<Dtype>>;
  friend class DagReverseIterator<PipeDag<Dtype>, true>;
public:
  PipeDag(
    std::shared_ptr<SegmentDag<Dtype>> segment_dag,
    std::shared_ptr<StageDag<Dtype>> stage_dag,
    PathType path_type,
    const std::string& name = "pipe_dag");
  ~PipeDag() {}

  std::vector<std::string> GetPipeNamesFromStageName(
    const std::string& stage_name) const;

  bool IsComputePipe(const std::string& pipe_name) const;
  std::string GetStageName(const std::string& pipe_name) const;

  BoxingInfo GetBoxingInfo(const std::string& boxing_name);

  StageStagePair GetStagePairFromNetPipe(const std::string& net_piep) const;

  int32_t GetThreadLocalId(const std::string& pipe_name) const;

  void AddDataNodeInBetween(DagNode* node1, DagNode* node2,
    const std::string& data_node_name);
private:
  // Internal class for managing the mapping between stage_name and pipe_name
  // Note that copy, boxing or net pipe does have corresponding stage.
  class StagePipeMap {
  public:
    StagePipeMap() = default;
    ~StagePipeMap() = default;

    void AddStagePipe(const std::string& stage_name,
      const std::string& pipe_name);
    const std::string& StageFromPipe(const std::string& pipe_name) const;
    const std::vector<std::string>& PipesFromStage(
      const std::string& stage_name) const;
  private:
    std::unordered_map<std::string, std::string> pipe_to_stage_;
    std::unordered_map<std::string, std::vector<std::string>> stage_to_pipes_;
  };

  // After add in_copy or out_copy, delegate the receiving connection of the 
  // compute pipe to its in_copy pipe. If have in_copy, return the in_copy, if
  // not have in_copy, return the pipe itself
  class CopyComputeMap {
  public:
    CopyComputeMap() = default;
    ~CopyComputeMap() = default;

    void AddInCopy(const std::string& in_copy_name,
      const std::string& compute_name);
    void AddOutCopy(const std::string& out_copy_name,
      const std::string& compute_name);
    std::string GetInCopyIfHave(const std::string& compute_name) const;
    std::string GetOutCopyIfHave(const std::string& compute_name) const;
    std::string GetComputeFromCopy(const std::string& copy_name) const;
  private:
    std::unordered_map<std::string, std::string> pipe_to_in_copy_;
    std::unordered_map<std::string, std::string> pipe_to_out_copy_;
    std::unordered_map<std::string, std::string> in_copy_to_pipe_;
    std::unordered_map<std::string, std::string> out_copy_to_pipe_;
  };

  using StageStageNet
    = std::unordered_map<std::string, std::unordered_map<std::string, std::string>>;
  // Manage the mapping between StageStagePair and the net pipe name inbetween
  class StageNetMap {
  public:
    StageNetMap() = default;
    ~StageNetMap() = default;

    void AddInNet(const std::string& from_stage, const std::string& to_stage,
      const std::string& in_net);
    void AddOutNet(const std::string& from_stage, const std::string& to_stage,
      const std::string& out_net);
    StageStagePair GetStagePairFromNetName(const std::string& net_name) const;
    std::unordered_map<std::string, std::string> GetOtherStageAndInNet(
      const std::string& stage_name) const;
    std::unordered_map<std::string, std::string> GetOtherStageAndOutNet(
      const std::string& stage_name) const;
  private:
    // Mapping from a net pipe name to the stage pair it serves
    std::unordered_map<std::string, StageStagePair> net_to_stage_pair_;
    // <A, <B, in_net>>
    // Memorize the in_net node in stage A, which connects to stage B.
    StageStageNet stage_to_stage_in_net_;
    // <A, <B, out_net>>
    // Memorize the out_net node in stage A, which connects to stage B.
    StageStageNet stage_to_stage_out_net_;
    
    void AddStagePairNet(const std::string& A, const std::string& B,
      const std::string& net_name, StageStageNet* stage_stage_net_tuple);
  };

  class StageBoxingMap {
  public:
    StageBoxingMap() = default;
    ~StageBoxingMap() = default;
    void AddStageAndInBoxing(
      const std::string& stage, const std::string& in_boxing);
    void AddStageAndOutBoxing(
      const std::string& stage, const std::string& out_boxing);
    bool HasInBoxing(const std::string& stage) const;
    std::string GetInBoxingFromStage(const std::string& stage) const;
    bool HasOutBoxing(const std::string& stage) const;
    std::string GetOutBoxingFromStage(const std::string& stage) const;
  private:
    std::unordered_map<std::string, std::string> stage_to_in_boxing_;
    std::unordered_map<std::string, std::string> stage_to_out_boxing_;
  };

private:
  std::shared_ptr<SegmentDag<Dtype>> segment_dag_;
  std::shared_ptr<StageDag<Dtype>> stage_dag_;

  void Build();

  StagePipeMap stage_pipe_map_;

  void ExpandStageToPipes();
  void SingleStageToSinglePipe(
    const std::string& stage_name,
    const std::string& segment_name,
    int32_t machine_id);
  void SingleStageToMultiplePipes(
    const std::string& stage_name,
    const std::string& segment_name,
    int32_t machine_id);
  // End of ExpandStageToPipes

  // Add in_copy and out_copy for compute pipe if necessary
  void AddCopyPipeNodes();
  void AddCopyPipeNodeForComputePipe(const std::string& pipe_name);
  CopyComputeMap copy_compute_map_;
  // End of AddCopyPipeNodes

  StageNetMap stage_net_map_;

  void AddNetPipeNodes();
  void AddNetPipeNode(
    bool is_in,
    const std::string& from_stage_name,
    int32_t from_machine_id,
    const std::string& to_stage_name,
    int32_t to_machine_id);
  void ConnectNetPipeNodes();
  // End of AddNetPipeNodes

  StageBoxingMap stage_boxing_map_;
  BoxingInfoMap boxing_info_map_;

  void AddBoxingPipeNodes();
  void AddBoxingInfos();
  OpNode<PipeMeta>* AddBoxingPipeNode(
    bool is_in,
    const std::string& bundle_name,
    int32_t stage_machine_id);
  bool NeedInBoxingPipe(
    const std::string& current_stage,
    int32_t current_machine_id);
  void AddInBoxingInfo(
    const std::string& in_boxing_name,
    const std::string& stage_name);
  bool NeedOutBoxingPipe(
    const std::string& current_stage,
    int32_t current_machine_id);
  void AddOutBoxingInfo(
    const std::string& out_boxing_name,
    const std::string& stage_name);

  std::vector<std::string> GetPrecedingSegmentNames(
    const std::string& stage_name) const;
  std::vector<std::string> GetSucceedingSegmentNames(
    const std::string& stage_name) const;

  std::vector<std::string> GetInDelegatePipesFromStage(
    const std::string& stage_name) const;
  std::vector<std::string> GetOutDelegatePipesFromStage(
    const std::string& stage_name) const;
  std::vector<std::string> GetDelegatePipesFromStage(
    const std::string& stage_name, bool in) const;
  // End of AddBoxingPipeNodes

  void ConnectOutgoingPipeNodes();
  void ConnectIncomingPipeNodes();

  void NotSameMachineOutgoing(
    const std::string& bundle_name,
    int32_t bundle_machine_id,
    const std::string& segment_name,
    const std::string& successor_bundle_name,
    int32_t successor_machine_id,
    const std::string& successor_segment_name);
  void SameMachineOutgoing(
    const std::string& bundle_name,
    int32_t bundle_machine_id,
    const std::string& segment_name,
    const std::string& successor_bundle_name,
    int32_t successor_machine_id,
    const std::string& successor_segment_name);
  void NotSameMachineIncoming(
    const std::string& predecessor_bundle_name,
    int32_t predecessor_machine_id,
    const std::string& predecessor_segment_name,
    const std::string& bundle_name,
    int32_t bundle_machine_id,
    const std::string& segment_name);
  // NOTE(jiyuan): We have no function for SameMachineIncoming()

  // Used for adding data nodes without successors (e.g., loss node)
  void AddDataNodesWithoutSuccessors();

  OpNode<PipeMeta>* AddOpNode(const std::string& pipe_name,
    int32_t thread_id, TaskType type);
  DataNode<EnvelopeMeta>* AddDataNode(const std::string& data_name);

  std::string build_pipe_name_with_local_id(
    const std::string &prefix,
    const std::string& stage_name,
    int32_t machine_id,
    int32_t local_id) const;

  std::string build_net_pipe_name(
    const std::string& prefix,
    const std::string& from_stage_name,
    const std::string& to_stage_name,
    int32_t from_machine_id,
    int32_t to_machine_id) const;

  std::string build_net_envelope_name(
    const std::string& prefix,
    const std::string& segment_envelope_name,
    int32_t from_machine_id,
    int32_t to_machine_id) const;

  std::string build_envelope_name(
    const std::string& prefix,
    const std::string& segment_envelope_name,
    int32_t machine_id,
    int32_t thread_local_id) const;

  std::string build_boxing_envelope_name(
    const std::string& prefix,
    const std::vector<std::string>& segment_envelope_names,
    int32_t machine_id,
    int32_t thread_local_id) const;
  std::string build_envelope_name_from_envelopes(
    const std::string& prefix,
    const std::vector<std::string>& segment_envelope_names,
    int32_t machine_id,
    int32_t local_id) const;

  const std::string compute_pipe_prefix_ = "";
  const std::string compute_envelope_prefix_ = "";

  const std::string in_copy_pipe_prefix_ = "in_copy_";
  const std::string out_copy_pipe_prefix_ = "out_copy_";
  const std::string in_copy_envelope_prefix_ = "in_copy_";
  const std::string out_copy_envelope_prefix_ = "out_copy_";

  const std::string in_boxing_pipe_prefix_ = "in_boxing_";
  const std::string out_boxing_pipe_prefix_ = "out_boxing_";
  const std::string in_boxing_envelope_prefix_ = "in_boxing_";
  const std::string out_boxing_envelope_prefix_ = "out_boxing_";

  const std::string in_net_pipe_prefix_ = "in_net_";
  const std::string out_net_pipe_prefix_ = "out_net_";
  const std::string on_net_envelope_prefix_ = "on_net_";
  const std::string to_net_envelope_prefix_ = "to_net_";
  const std::string from_net_envelope_prefix_ = "from_net_";

  PipeDag(const PipeDag& other) = delete;
  PipeDag& operator=(const PipeDag& other) = delete;
};
}  // namespace caffe
#endif  // _DAG_PIPE_DAG_H_
