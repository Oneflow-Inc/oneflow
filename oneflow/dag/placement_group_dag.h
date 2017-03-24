#ifndef _DAG_PLACEMENT_GROUP_DAG_H_
#define _DAG_PLACEMENT_GROUP_DAG_H_
#include <vector>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "dag/dag_node.h"
#include "dag/dag.h"
#include "context/placement_info.h"
#include "dag/dag_builder.cpp"
#include "dag/dag_iterator.h"
/*
PlacementGroupDag
It is used to specify the binding relation between layers in LogicalDag and
devices. The DAG is built according to strategy proto, each OpNode corresponding
to a PlacementGroup field in strategy proto.
Note:
(1) The PlacementGroups are declared in topological order;
(2) The layers in a PlacementGroup should form a connected component (i.e., some
continuous layers in LogicalDag;
(3) The layers in a PlacementGroup should be declared in topological order;
(4) The PlacementInfo in each PlacementGroup specifies the device set and parallel
policy;
(5) Some PlacementGroup's PlacementInfo is not declared, however it can be
inferred from other neighboring PlacementGroup's PlacementInfo;
(6) The PlacementGroupDag will fill the PlacementInfo of each layer in LogicalDag;
*/
namespace oneflow {
class BlobMeta;

class PlacementGroupMeta;

/*
template<typename DAG, bool isconst = false>
class DagIterator;

template<typename DAG, bool isconst = false> 
class DagReverseIterator;
*/

template <typename Dtype>
class LogicalDag;

class ConfigParser;

class StrategyDescriptor;

template <typename Dtype>
class PlacementGroupDag : public Dag<BlobMeta, PlacementGroupMeta> {
  friend class DagIterator<PlacementGroupDag<Dtype>>;
  friend class DagIterator<PlacementGroupDag<Dtype>, true>;
  friend class DagReverseIterator<PlacementGroupDag<Dtype>>;
  friend class DagReverseIterator<PlacementGroupDag<Dtype>, true>;
  public:
    PlacementGroupDag(std::shared_ptr<LogicalDag<Dtype>> logical_dag,
      std::shared_ptr<StrategyDescriptor> strategy_descriptor,
      PathType path_type,
      const std::string& name = "placement_group_dag");
    ~PlacementGroupDag() {}

    std::vector<std::string> GetLayerNamesInGroup(
      const std::string& group_name) const;

  private:
    std::shared_ptr<LogicalDag<Dtype>> logical_dag_;
    std::shared_ptr<StrategyDescriptor> strategy_descriptor_;

    void VerifyPreConditions();
    void Build();
    void AddOpNodes();
    ONode* AddOpNode(
      const std::string& op_name, const PlacementInfo& placement_info);
    DNode* AddDataNode(const std::string& blob_name);
    void AddAndConnectDataNodes();
    void SetPlacementInfoForLayer();

    // Ensure:
    // (1) The PlacementGroups are declared in topological order;
    // (2) The layers in each PlacementGroups are declared in topological order;
    // (3) The layer name in PlacementGroup is consistent with that in LogicalDag;
    // (4) Each layer in LogicalDag is exactly declared once in PlacementGroups;
    void VerifyDependency();

    // Some particular groups have specific constraints, such as:
    // (1) The parallel policy of 'data' group can be un-specified
    // (kUnknownParallel) or be kDataParallel. If it is not specified, it will
    // be set as kDataParallel; The 'softmax' and 'loss' group have the same
    // requirements.
    // (2) The device set of 'data' group can be either not specified or with the
    // same device set as its successor in the lowest level value;
    // (3) The device set of 'softmax' group can be either specified or be with
    // the same device set as its predecessor in highest level value;
    // (4) The 'loss' group will inherit the device set from its predecessor
    // in the highest level value, such as 'softmax' group;
    void CompleteGroupInfo();

    // The group with kModelParallel policy must have exactly one layer. Group
    // with more than one layers can not have kModelParallel policy.
    // deprecated: since group being ModelParallel may have more than one layers,
    // if t has elem-wise layer.
    // void VerifyGroupInModelParallel();

    // Elem-wise layer must have the same PlacementInfo with its preceding layer
    void VerifyElemWiseLayerDependency();

    void CompleteGroupInfoForwardDependency(int32_t gid);
    void CompleteGroupInfoBackwardDependency(int32_t gid);

    PlacementGroupDag(const PlacementGroupDag& other) = delete;
    PlacementGroupDag& operator=(const PlacementGroupDag& other) = delete;
};
}  // namespace oneflow
#endif  // _DAG_PLACEMENT_GROUP_DAG_H_
