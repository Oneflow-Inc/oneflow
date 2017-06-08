#include "oneflow/core/graph/comp_task_node.h"
#include "oneflow/core/graph/model_update_task_graph.h"
#include "oneflow/core/graph/model_save_task_graph.h"
#include "oneflow/core/operator/operator_manager.h"
#include "oneflow/core/operator/clone_op.h"

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
  return IDMgr::Singleton().MachineName4MachineId(stage_node()->machine_id())
      + ":"
      + std::to_string(IDMgr::Singleton().DevPhyId4ThrdLocId(thrd_loc_id()));
}

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec) {
  std::sort(comp_node_vec->begin(), comp_node_vec->end(), []
      (const CompTaskNode* lhs, const CompTaskNode* rhs) {
    return lhs->parallel_id() < rhs->parallel_id();
  });
}

} // namespace oneflow
