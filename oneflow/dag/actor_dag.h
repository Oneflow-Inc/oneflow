#ifndef _DAG_ACTOR_DAG_H_
#define _DAG_ACTOR_DAG_H_
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include "dag/dag_node.h"
#include "dag/node_meta.h"
#include "dag/dag.h"
#include "common/string_pair.h"
#include "dag/boxing_info.h"
/*
ActorDag consists of a forward Dag (basically cloned from PipeDag) and a 
backward Dag (if necessary).
*/
namespace oneflow {
enum class TaskType;

class EventMeta;

class ActorMeta;

/*
template <typename DAG, bool isconst = false>
class DagIterator;

template <typename DAG, bool isconst = false>
class DagReverseIterator;
*/

template <typename Dtype>
class LogicalDag;

template <typename Dtype>
class SegmentDag;

template <typename Dtype>
class StageDag;

template <typename Dtype>
class PipeDag;

using SegmentSegmentPair = StringPair;

template <typename Dtype>
class ActorDag : public Dag<EventMeta, ActorMeta> {
  friend class DagIterator<ActorDag<Dtype>>;
  friend class DagIterator<ActorDag<Dtype>, true>;
  friend class DagReverseIterator<ActorDag<Dtype>>;
  friend class DagReverseIterator<ActorDag<Dtype>, true>;
public:
  ActorDag(
    std::shared_ptr<LogicalDag<Dtype>> logical_dag,
    std::shared_ptr<SegmentDag<Dtype>> segment_dag,
    std::shared_ptr<StageDag<Dtype>> stage_dag,
    std::shared_ptr<PipeDag<Dtype>> pipe_dag,
    PathType path_type,
    const std::string& name = "actor_dag");
  ~ActorDag() {}

  BoxingInfo GetForwardBoxingInfo(const std::string& boxing_actor_name);

  bool has_BP() const { return has_BP_; }

  // The following functions only work for kComputeTask or kDataTask type actors.
  // Some of actor does not have corresponding segment or stage.
  std::vector<std::string> GetLayerNamesFromActor(
    const std::string& actor_name) const;

  std::string GetSegmentNameFromActor(
    const std::string& actor_name) const;

  std::string GetStageNameFromActor(
    const std::string& actor_name) const;

  std::string GetPipeNameFromActor(
    const std::string& actor_name) const;
  // End

  std::string GetForwardTaskName(const std::string& backward_task_name) const;
  std::string GetBackwardTaskName(const std::string& forward_task_name) const;

  std::string GetForwardActorFromPipe(const std::string& pipe_name) const;
  std::string GetBackwardActorFromPipe(const std::string& pipe_name) const;

  int32_t GetTaskID(const std::string& actor_name) const;

  std::string GetFirstDescendantComputeNodeName(
    const std::string& op_name) const;

private:
  class ActorPipeMap {
  public:
    ActorPipeMap() = default;
    ~ActorPipeMap() = default;
    void AddForwardActorPipe(const std::string& forward_actor,
      const std::string& pipe);
    void AddBackwardActorPipe(const std::string& backward_actor,
      const std::string& pipe);
    std::string GetPipeFromActor(const std::string& actor) const;
    std::string GetBackwardFromForward(const std::string& foward_actor) const;
    std::string GetForwardFromBackward(const std::string& backward_actor) const;
    std::string GetForwardFromPipe(const std::string& pipe) const;
    std::string GetBackwardFromPipe(const std::string& pipe) const;
  private:
    std::unordered_map<std::string, std::string> forward_actor_to_pipe_;
    std::unordered_map<std::string, std::string> pipe_to_forward_actor_;
    std::unordered_map<std::string, std::string> backward_actor_to_pipe_;
    std::unordered_map<std::string, std::string> pipe_to_backward_actor_;
  };

private:
    std::shared_ptr<LogicalDag<Dtype>> logical_dag_;
    std::shared_ptr<SegmentDag<Dtype>> segment_dag_;
    std::shared_ptr<StageDag<Dtype>> stage_dag_;
    std::shared_ptr<PipeDag<Dtype>> pipe_dag_;
    bool has_BP_;

    void Build();

    std::unordered_set<std::string> last_pipes_in_bp_;
    void InferHasBpForPipeDag();

    void ForwardBuildDag();
    void ForwardAddActorNodes();
    void ForwardConnectActorNodes();

    void ForwardUpdateBoxingInfo(const std::string& boxing_pipe_name);

    ActorPipeMap actor_pipe_map_;
    BoxingInfoMap boxing_info_map_;

    void BackwardBuildDag();
    void BackwardAddActorNodes();
    void ConnectTurningPointAtLossNode();
    void BackwardConnectActorNodes();

    std::vector<std::string> loss_pipes_;
    // The backward and forward actors corresponding to the same pipe node share
    // the same layers. The layers are configured and created in forward pass. 
    // The BoxingToMeta is only needed while configuring the boxing layer in 
    // forward pass. Therefore, we don't need BoxingToMeta while building 
    // backward DAG.

    OpNode<ActorMeta>* AddOpNode(
      const std::string& actor_name,
      int32_t task_id,
      TaskType type);
    DataNode<EventMeta>* AddDataNode(const std::string& data_name);
    bool HasOpNode(const std::string& actor_name) const;

    const std::string forward_prefix_ = "forward_";
    const std::string backward_prefix_ = "backward_";

    std::string build_actor_name(const std::string& prefix,
      const std::string& pipe_name) const;
    std::string build_event_name(const std::string& prefix,
      const std::string& envelope_name) const;

    ActorDag(const ActorDag& other) = delete;
    ActorDag& operator=(const ActorDag& other) = delete;
};
}  // namespace oneflow
#endif  // _DAG_ACTOR_DAG_H_
