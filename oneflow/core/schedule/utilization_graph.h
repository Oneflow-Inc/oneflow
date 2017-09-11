#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/utilization.h"

namespace oneflow {
namespace schedule {

class UtilizationGraph final {
#define UTILIZATION_ARC_SEQ                                                  \
  OF_PP_MAKE_TUPLE_SEQ(ComputationUtilization, DeviceComputationUtilization) \
  OF_PP_MAKE_TUPLE_SEQ(DeviceComputationUtilization, StreamUtilization)      \
  OF_PP_MAKE_TUPLE_SEQ(DeviceComputationUtilization, TaskUtilization)        \
  OF_PP_MAKE_TUPLE_SEQ(StreamUtilization, TaskStreamUtilization)             \
  OF_PP_MAKE_TUPLE_SEQ(TaskUtilization, TaskStreamUtilization)               \
  OF_PP_MAKE_TUPLE_SEQ(MemoryUtilization, DeviceMemoryUtilization)           \
  OF_PP_MAKE_TUPLE_SEQ(DeviceMemoryUtilization, RegstDescUtilization)        \
  OF_PP_MAKE_TUPLE_SEQ(RegstDescUtilization, RegstUtilization)

 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationGraph);
  explicit UtilizationGraph(const SGraph* sgraph) : sgraph_(sgraph) {
    InitRoot();
  }
  ~UtilizationGraph() = default;

  void ForEachUtilization(const std::function<void(Utilization*)>& cb) const;
  Utilization* FindOrCreateUtilization(const UtilizationResource& resource);
  void Connect(Utilization* src, Utilization* dst);
  void ForEachUtilizationInPath(Utilization* leaf,
                                const std::function<void(Utilization*)>& cb);

  //	getter
  inline const SGraph* sgraph() const { return sgraph_; }
  inline const ComputationUtilization& computation() const {
    return *computation_;
  }
  inline const MemoryUtilization& memory() const { return *memory_; }

  template<typename U>
  inline const NodeMgr<U>& node_mgr() const;

  template<typename src_node_type, typename dst_node_type>
  inline const ArcMgr<Arc<src_node_type, dst_node_type>>& arc_mgr() const;

  inline const ArcMgr<Arc<Utilization, Utilization>>& utilization_arc_mgr()
      const {
    return utilization_arc_mgr_;
  }

  inline const ArcMgr<Arc<Utilization, Utilization>>& inner2leaf_arc_mgr()
      const {
    return inner2leaf_arc_mgr_;
  }

  //	setter

  template<typename U>
  inline NodeMgr<U>* mut_node_mgr();

  template<typename src_node_type, typename dst_node_type>
  inline ArcMgr<Arc<src_node_type, dst_node_type>>* mut_arc_mgr();

  inline ArcMgr<Arc<Utilization, Utilization>>* mut_utilization_arc_mgr() {
    return &utilization_arc_mgr_;
  }
  inline ArcMgr<Arc<Utilization, Utilization>>* mut_inner2leaf_arc_mgr() {
    return &inner2leaf_arc_mgr_;
  }

  inline HashMap<UtilizationProto*, std::unique_ptr<UtilizationProto>>*
  mut_utilization_proto_store() {
    return &utilization_proto_store_;
  }

 private:
  void InitRoot();

  Utilization* FindUtilization(const UtilizationResource& resource) const;
  template<typename U>
  U* FindConcreteUtilization(const UtilizationResource& resource) const;

  Utilization* CreateUtilization(const UtilizationResource& resource);
  template<typename U>
  U* CreateConcreteUtilization(const UtilizationResource& resource);

  template<typename src_utilization, typename dst_utilization>
  void ConnectConcreteArc(Utilization* src, Utilization* dst);

  const SGraph* sgraph_;
  ComputationUtilization* computation_;
  MemoryUtilization* memory_;
  HashMap<UtilizationProto*, std::unique_ptr<UtilizationProto>>
      utilization_proto_store_;

#define UTILIZATION_NODE_MGR_MEMBER(field, class_name) \
  NodeMgr<class_name> OF_PP_CAT(class_name, _node_mgr_);
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_NODE_MGR_MEMBER, UTILIZATION_TYPE_SEQ);

#define UTILIZATION_ARC_MGR_MEMBER(src_node_type, dst_node_type) \
  ArcMgr<Arc<src_node_type, dst_node_type>>                      \
      src_node_type##_##dst_node_type##_arc_mgr_;
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_ARC_MGR_MEMBER, UTILIZATION_ARC_SEQ);
  ArcMgr<Arc<Utilization, Utilization>> utilization_arc_mgr_;
  ArcMgr<Arc<Utilization, Utilization>> inner2leaf_arc_mgr_;
};

//	getter
#define SPECIALIZE_UGRAPH_NODE_MGR_GETTER(type_case, class_name)             \
  template<>                                                                 \
  inline const NodeMgr<class_name>& UtilizationGraph::node_mgr<class_name>() \
      const {                                                                \
    return OF_PP_CAT(class_name, _node_mgr_);                                \
  }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_UGRAPH_NODE_MGR_GETTER, UTILIZATION_TYPE_SEQ)

#define UTILIZATION_ARC_MGR_GETTER(src_node_type, dst_node_type)    \
  template<>                                                        \
  inline const ArcMgr<Arc<src_node_type, dst_node_type>>&           \
  UtilizationGraph::arc_mgr<src_node_type, dst_node_type>() const { \
    return src_node_type##_##dst_node_type##_arc_mgr_;              \
  }
OF_PP_FOR_EACH_TUPLE(UTILIZATION_ARC_MGR_GETTER, UTILIZATION_ARC_SEQ);

//	setter
#define SPECIALIZE_UGRAPH_NODE_MGR_SETTER(type_case, class_name)             \
  template<>                                                                 \
  inline NodeMgr<class_name>* UtilizationGraph::mut_node_mgr<class_name>() { \
    return &OF_PP_CAT(class_name, _node_mgr_);                               \
  }
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_UGRAPH_NODE_MGR_SETTER, UTILIZATION_TYPE_SEQ)

#define UTILIZATION_ARC_MGR_SETTER(src_node_type, dst_node_type)  \
  template<>                                                      \
  inline ArcMgr<Arc<src_node_type, dst_node_type>>*               \
  UtilizationGraph::mut_arc_mgr<src_node_type, dst_node_type>() { \
    return &src_node_type##_##dst_node_type##_arc_mgr_;           \
  }
OF_PP_FOR_EACH_TUPLE(UTILIZATION_ARC_MGR_SETTER, UTILIZATION_ARC_SEQ);

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_
