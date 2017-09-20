#include "oneflow/core/schedule/visualization.h"
#include "oneflow/core/schedule/sxml.h"
namespace oneflow {
namespace schedule {

std::string Visualization::UGraph2TaskSVGString(
    const UtilizationGraph& ugraph) const {
  int task_index = 0;
  std::list<const TaskUtilization*> tasks;
  ugraph.node_mgr<TaskUtilization>().ForEach(
      [&](const TaskUtilization& task) { tasks.push_back(&task); });
  float start_at = ugraph.computation().utilization_proto().start_at();
  float end_at = ugraph.computation().utilization_proto().end_at();
  float duration = end_at - start_at;
  // clang-format off
	SXML svg{"svg", {
		{"@", {
				{"width", "100%"},
				{"xmlns", "http://www.w3.org/2000/svg"},
				{"xmlns:xlink", "http://www.w3.org/1999/xlink"},
		}},
		{"", SXML::List(tasks, [&](const TaskUtilization* task){
			SXML svg{"", SXML::List(task->raw_protos(), [&](const UtilizationProto* u){
				float x = (u->start_at() - start_at) / duration;
				float y = 50 * task_index;
				float w = (u->end_at() - u->start_at()) / duration;
				float h = 50;
				return SXML{"rect", {
					{"@", {
						{"x", std::to_string(x * 100) + "%"},
						{"y", std::to_string(y) + "px"},
						{"width", std::to_string(w * 100) + "%"},
						{"height", std::to_string(h) + "px"},
						{"stroke", "green"},
						{"fill-opacity", 1}}}
				}};
			})};
			++ task_index;
			return svg;
		})}
	}};
  // clang-format on
  return "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
         + svg.ToString();
}

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
