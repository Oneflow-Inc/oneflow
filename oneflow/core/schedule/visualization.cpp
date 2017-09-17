#include "oneflow/core/schedule/visualization.h"
namespace oneflow {
namespace schedule {

std::string Visualization::UGraph2DotString(
    const UtilizationGraph& ugraph) const {
  std::stringstream ss;
  auto render_node_pair = [&](const Utilization& src, const Utilization& dst) {
    ss << "\t\"" << src.VisualStr() << "\" -> \"" << dst.VisualStr() << "\";"
       << std::endl;
  };
  auto for_each_memory_utilization_arc =
      [&](const std::function<void(const Arc<Utilization, Utilization>&)>& cb) {
        std::list<const Utilization*> l;
        auto collect = [&](const Utilization& node) { l.push_back(&node); };
        ugraph.node_mgr<MemoryUtilization>().ForEach(collect);
        ugraph.node_mgr<DeviceMemoryUtilization>().ForEach(collect);
        ugraph.node_mgr<RegstDescUtilization>().ForEach(collect);
        ugraph.utilization_arc_mgr().InputArc(l, cb);
      };
  auto for_each_computation_utilization_arc =
      [&](const std::function<void(const Arc<Utilization, Utilization>&)>& cb) {
        std::list<const Utilization*> l;
        auto collect = [&](const Utilization& node) { l.push_back(&node); };
        ugraph.node_mgr<ComputationUtilization>().ForEach(collect);
        ugraph.node_mgr<DeviceComputationUtilization>().ForEach(collect);
        ugraph.node_mgr<StreamUtilization>().ForEach(collect);
        ugraph.utilization_arc_mgr().InputArc(l, cb);
      };
  auto for_each_stream_task_arc =
      [&](const std::function<void(const Utilization&, const Utilization&)>&
              cb) {
        ugraph.node_mgr<TaskStreamUtilization>().ForEach(
            [&](const TaskStreamUtilization& node) {
              const StreamUtilization* src;
              const TaskUtilization* dst;
              ugraph.arc_mgr<StreamUtilization, TaskStreamUtilization>().Input(
                  &node, &src);
              ugraph.arc_mgr<TaskUtilization, TaskStreamUtilization>().Input(
                  &node, &dst);
              CHECK(src);
              CHECK(dst);
              cb(*src, *dst);
            });
      };
  std::unordered_map<uint64_t, const Utilization*> task_id2task;
  std::unordered_map<uint64_t, const Utilization*> regst_desc_id2regst_desc;
  ugraph.node_mgr<RegstDescUtilization>().ForEach(
      [&](const RegstDescUtilization& node) {
        regst_desc_id2regst_desc[node.regst_desc_id()] = &node;
      });
  ugraph.node_mgr<TaskUtilization>().ForEach([&](const TaskUtilization& node) {
    task_id2task[node.task_id()] = &node;
  });
  auto for_each_task_regst_desc_arc =
      [&](const std::function<void(const Utilization&, const Utilization&)>&
              cb) {
        ugraph.sgraph().node_mgr<SRegstDesc>().ForEach(
            [&](const SRegstDesc& regst_desc) {
              const Utilization* src =
                  task_id2task[regst_desc.owner_task().id()];
              const Utilization* dst =
                  regst_desc_id2regst_desc[regst_desc.id()];
              CHECK(src);
              CHECK(dst);
              cb(*src, *dst);
            });
      };
  ss << "digraph {" << std::endl;
  ss << "\t rankdir=\"LR\";" << std::endl;
  for_each_computation_utilization_arc(
      [&](const Arc<Utilization, Utilization>& arc) {
        render_node_pair(*arc.src_node(), *arc.dst_node());
      });
  for_each_stream_task_arc(render_node_pair);
  for_each_task_regst_desc_arc(render_node_pair);
  for_each_memory_utilization_arc(
      [&](const Arc<Utilization, Utilization>& arc) {
        render_node_pair(*arc.dst_node(), *arc.src_node());
      });
  ss << "}" << std::endl;
  return ss.str();
}

}  // namespace schedule
}  // namespace oneflow
