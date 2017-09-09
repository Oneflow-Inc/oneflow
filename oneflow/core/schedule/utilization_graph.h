#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/utilization.h"

namespace oneflow {
namespace schedule {

class UtilizationGraph final {
#define UTILIZATION_NODE_SEQ                                              \
  OF_PP_MAKE_TUPLE_SEQ(dev_computation_mgr, DeviceComputationUtilization) \
  OF_PP_MAKE_TUPLE_SEQ(stream_mgr, StreamUtilization)                     \
  OF_PP_MAKE_TUPLE_SEQ(task_mgr, TaskUtilization)                         \
  OF_PP_MAKE_TUPLE_SEQ(task_stream_mgr, TaskStreamUtilization)            \
  OF_PP_MAKE_TUPLE_SEQ(dev_memory_mgr, DeviceMemoryUtilization)           \
  OF_PP_MAKE_TUPLE_SEQ(regst_desc_mgr, RegstDescUtilization)              \
  OF_PP_MAKE_TUPLE_SEQ(regst_mgr, RegstUtilization)

#define UTILIZATION_DIRECT_ARC_SEQ                                             \
  OF_PP_MAKE_TUPLE_SEQ(root2dc_arc_mgr, ComputationUtilization,                \
                       DeviceComputationUtilization)                           \
  OF_PP_MAKE_TUPLE_SEQ(dc2s_arc_mgr, DeviceComputationUtilization,             \
                       StreamUtilization)                                      \
  OF_PP_MAKE_TUPLE_SEQ(dc2t_arc_mgr, DeviceComputationUtilization,             \
                       TaskUtilization)                                        \
  OF_PP_MAKE_TUPLE_SEQ(s2ts_arc_mgr, StreamUtilization, TaskStreamUtilization) \
  OF_PP_MAKE_TUPLE_SEQ(t2ts_arc_mgr, TaskUtilization, TaskStreamUtilization)   \
  OF_PP_MAKE_TUPLE_SEQ(root2dm_arc_mgr, MemoryUtilization,                     \
                       DeviceMemoryUtilization)                                \
  OF_PP_MAKE_TUPLE_SEQ(dm2rd_arc_mgr, DeviceMemoryUtilization,                 \
                       RegstDescUtilization)                                   \
  OF_PP_MAKE_TUPLE_SEQ(rd2r_arc_mgr, RegstDescUtilization, RegstUtilization)

#define UTILIZATION_LEAF_ARC_SEQ                               \
  OF_PP_MAKE_TUPLE_SEQ(c2leaf_arc_mgr, ComputationUtilization, \
                       TaskStreamUtilization)                  \
  OF_PP_MAKE_TUPLE_SEQ(m2leaf_arc_mgr, MemoryUtilization, RegstUtilization)

#define UTILIZATION_ARC_SEQ UTILIZATION_DIRECT_ARC_SEQ UTILIZATION_LEAF_ARC_SEQ

 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationGraph);
  explicit UtilizationGraph(const SGraph* sgraph) : sgraph_(sgraph) {}
  ~UtilizationGraph() = default;

  void ForEachUtilization(const std::function<void(Utilization*)>& cb) const;

  //	getter
  inline const SGraph* sgraph() const { return sgraph_; }
  inline const ComputationUtilization& computation() const {
    return computation_;
  }
  inline const MemoryUtilization& memory() const { return memory_; }
#define UTILIZATION_NODE_MGR_GETTER(field, class_name) \
  inline const NodeMgr<class_name>& field() const {    \
    return OF_PP_CAT(field, _);                        \
  }
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_NODE_MGR_GETTER, UTILIZATION_NODE_SEQ)

#define UTILIZATION_ARC_MGR_GETTER(field, src_node_type, dst_node_type)   \
  inline const ArcMgr<Arc<src_node_type, dst_node_type>>& field() const { \
    return OF_PP_CAT(field, _);                                           \
  }
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_ARC_MGR_GETTER, UTILIZATION_ARC_SEQ);

//	setter
#define UTILIZATION_NODE_MGR_SETTER(field, class_name)   \
  inline NodeMgr<class_name>* OF_PP_CAT(mut_, field)() { \
    return &OF_PP_CAT(field, _);                         \
  }
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_NODE_MGR_SETTER, UTILIZATION_NODE_SEQ)

#define UTILIZATION_ARC_MGR_SETTER(field, src_node_type, dst_node_type)        \
  inline ArcMgr<Arc<src_node_type, dst_node_type>>* OF_PP_CAT(mut_, field)() { \
    return &OF_PP_CAT(field, _);                                               \
  }
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_ARC_MGR_SETTER, UTILIZATION_ARC_SEQ);

  inline HashMap<UtilizationProto*, std::unique_ptr<UtilizationProto>>*
  mut_utilization_proto_store() {
    return &utilization_proto_store_;
  }

 private:
  const SGraph* sgraph_;
  ComputationUtilization computation_;
  MemoryUtilization memory_;
  HashMap<UtilizationProto*, std::unique_ptr<UtilizationProto>>
      utilization_proto_store_;

#define UTILIZATION_NODE_MGR_MEMBER(field, class_name) \
  NodeMgr<class_name> OF_PP_CAT(field, _);
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_NODE_MGR_MEMBER, UTILIZATION_NODE_SEQ);

#define UTILIZATION_ARC_MGR_MEMBER(field, src_node_type, dst_node_type) \
  ArcMgr<Arc<src_node_type, dst_node_type>> OF_PP_CAT(field, _);
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_ARC_MGR_MEMBER, UTILIZATION_ARC_SEQ);
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_
