#include "oneflow/core/schedule/visualization.h"
namespace oneflow {
namespace schedule {

std::string Visualization::UGraph2DotString(
    const UtilizationGraph& ugraph) const {
  std::stringstream ss;
  ss << "digraph {" << std::endl;
  ss << "\t rankdir=\"LR\";" << std::endl;
  ugraph.utilization_arc_mgr().ForEach(
      [&](const Arc<Utilization, Utilization>& arc) {
        if (!ugraph.utilization_arc_mgr().Output(arc.dst_node())) return;
        ss << "\t\"" << arc.src_node()->VisualStr() << "\" -> \""
           << arc.dst_node()->VisualStr() << "\";" << std::endl;
      });
  ss << "}" << std::endl;
  return ss.str();
}

}  // namespace schedule
}  // namespace oneflow
