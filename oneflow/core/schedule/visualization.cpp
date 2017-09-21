#include "oneflow/core/schedule/visualization.h"
#include "oneflow/core/schedule/sxml.h"
namespace oneflow {
namespace schedule {

std::string Visualization::UGraph2TaskSVGString(
    const UtilizationGraph& ugraph) const {
  float start_at = ugraph.computation().utilization_proto().start_at();
  float end_at = ugraph.computation().utilization_proto().end_at();
  float duration = end_at - start_at;
  float padding = 5;
  float row_height = 40;

  int stream_index = 0;
  std::list<const TaskStreamUtilization*> tasks;
  ugraph.node_mgr<TaskStreamUtilization>().ForEach(
      [&](const TaskStreamUtilization& task) { tasks.push_back(&task); });

  int task_index = 0;
  std::list<const StreamUtilization*> streams;
  ugraph.node_mgr<StreamUtilization>().ForEach(
      [&](const StreamUtilization& stream) { streams.push_back(&stream); });
  auto get_x = [&](const UtilizationProto* u) {
    float x = (u->start_at() - start_at) / duration * 100;
    return std::to_string(x) + "%";
  };
  auto get_w = [&](const UtilizationProto* u) {
    float x = (u->end_at() - u->start_at()) / duration * 100;
    return std::to_string(x) + "%";
  };
  int task_height = row_height * tasks.size();
  int stream_height = row_height * streams.size();
  // clang-format off
	SXML svg{"svg", {
		{"@", {
			{"width", 1928},
			{"height", task_height + row_height * 3 + stream_height},
			{"xmlns", "http://www.w3.org/2000/svg"},
			{"xmlns:xlink", "http://www.w3.org/1999/xlink"},
		}},
		{"svg", {
			{"@", {
				{"x", "4%"},
				{"y", 0},
				{"width", "92%"},
				{"height", "100%"},
			}},
			{"svg", {
				{"@", {
					{"x", 0},
					{"y", 0},
					{"width", "100%"},
					{"height", row_height},
				}},
				{"text", {
					{"@", {
						{"x", 0},
						{"y", 25},
						{"font-size", 20},
					}},
					{"", "Stream"},
				}},
			}},
			{"svg", {
				{"@", {
					{"x", 0},
					{"y", stream_height + row_height},
					{"width", "100%"},
					{"height", row_height},
				}},
				{"text", {
					{"@", {
						{"x", 0},
						{"y", 25},
						{"font-size", 20},
					}},
					{"", "Task Stream"},
				}},
			}},
			{"svg", {
				{"@", {
					{"x", 0},
					{"y", row_height},
					{"width", "100%"},
					{"height", stream_height},
				}},
				{"", SXML::List(streams, [&](const StreamUtilization* stream){
					SXML svg{"svg", {
						{"@", {
							{"x", "0%"},
							{"y", stream_index * row_height},
							{"width", "100%"},
							{"height", row_height},
						}},
						{"svg", {
							{"@", {
								{"x", "0%"},
								{"y", 0},
								{"width", "10%"},
								{"height", "100%"},
							}},
							{"text", {
								{"@", {
									{"x", 0},
									{"y", 13},
									{"font-size", 13},
								}},
								{"", "device-id: " + std::to_string(stream->device_id())},
							}},
							{"text", {
								{"@", {
									{"x", 0},
									{"y", 30},
									{"font-size", 13},
								}},
								{"", "stream-id: " + std::to_string(stream->stream_id())},
							}},
						}},
						{"svg", {
							{"@", {
								{"x", "10%"},
								{"y", 0},
								{"width", "90%"},
								{"height", "100%"},
							}},
							{"rect", {
								{"@", {
									{"x", 0},
									{"y", padding},
									{"width", "100%"},
									{"height", row_height - padding * 2},
									{"stroke", "black"},
									{"stroke-width", 1},
									{"fill-opacity", 1},
									{"fill", "white"},
								}}
							}},
							{"", SXML::List(stream->raw_protos(), [&](const UtilizationProto* u){
								return SXML{"rect", {
									{"@", {
										{"x", get_x(u)},
										{"y", padding},
										{"width", get_w(u)},
										{"height", row_height - padding * 2},
										{"stroke", "green"},
										{"fill", "blue"}}}
								}};
							})}
						}},
					}};
					++ stream_index;
					return svg;
				})}
			}},
			{"svg", {
				{"@", {
					{"x", 0},
					{"y", row_height * 2 + stream_height},
					{"width", "100%"},
					{"height", task_height},
				}},
				{"", SXML::List(tasks, [&](const TaskStreamUtilization* task){
					SXML svg{"svg", {
						{"@", {
							{"x", "0%"},
							{"y", task_index * row_height},
							{"width", "100%"},
							{"height", row_height},
						}},
						{"svg", {
							{"@", {
								{"x", "0%"},
								{"y", 0},
								{"width", "10%"},
								{"height", "100%"},
							}},
							{"text", {
								{"@", {
									{"x", 0},
									{"y", 13},
									{"font-size", 13},
								}},
								{"", "task-id: " + std::to_string(task->task_id())},
							}},
							{"text", {
								{"@", {
									{"x", 0},
									{"y", 30},
									{"font-size", 13},
								}},
								{"", "stream-id: " + std::to_string(task->stream_id())},
							}},
						}},
						{"svg", {
							{"@", {
								{"x", "10%"},
								{"y", 0},
								{"width", "90%"},
								{"height", "100%"},
							}},
							{"rect", {
								{"@", {
									{"x", 0},
									{"y", padding},
									{"width", "100%"},
									{"height", row_height - padding * 2},
									{"stroke", "black"},
									{"stroke-width", 1},
									{"fill-opacity", 1},
									{"fill", "white"},
								}}
							}},
							{"", SXML::List(task->raw_protos(), [&](const UtilizationProto* u){
								return SXML{"rect", {
									{"@", {
										{"x", get_x(u)},
										{"y", padding},
										{"width", get_w(u)},
										{"height", row_height - padding * 2},
										{"stroke", "green"},
										{"fill", "blue"}}}
								}};
							})}
						}},
					}};
					++ task_index;
					return svg;
				})}
			}},
			
		}},
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
