#include "graph/comp_task_node.h"
#include "graph/model_update_task_graph.h"
#include "graph/model_save_task_graph.h"
#include "operator/operator_manager.h"
#include "operator/clone_op.h"

namespace oneflow {

std::string CompTaskNode::VisualStr() const {
  std::stringstream ss;
  ss << TaskNode::VisualStr() 
     << "Compute" << ":"
     << stage_node()->machine_id_str() << ":"
     << thrd_loc_id_str() << "\\n"
     << chain_node()->VisualStr();
  return ss.str();
}

std::string CompTaskNode::device_name() const {
  TODO();
}

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec) {
  std::sort(comp_node_vec->begin(), comp_node_vec->end(), []
      (const CompTaskNode* lhs, const CompTaskNode* rhs) {
    return lhs->parallel_id() < rhs->parallel_id();
  });
}

} // namespace oneflow
