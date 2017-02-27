#include "dag/segment_dag.h"
#include "glog/logging.h"
#include <list>

namespace oneflow {

struct Segment {
  // Main
  std::vector<const LogicalOpNode*> op_nodes;
  // Attached Properties
  std::unordered_set<const LogicalDataNode*> predecessors;
  std::unordered_set<const LogicalDataNode*> successors;
  std::unordered_set<const LogicalOpNode*> ancestors;
  std::unordered_set<const LogicalOpNode*> descendants;
};

static void InitSegments(
    const LogicalDag* logical_dag,
    std::list<Segment>* segment_list,
    std::unordered_map<const LogicalOpNode*,
                       std::list<Segment>::iterator>* node2segment) {
  // Init one Segment with one OpNode
  segment_list->clear();
  node2segment->clear();
  // Init layers
  for (const std::unique_ptr<OpNode>& op_node : logical_dag->op_node_vec()) {
    auto logical_opnode = of_dynamic_cast<const LogicalOpNode*>(op_node.get());
    segment_list->emplace_back();
    node2segment->insert({logical_opnode, --segment_list->end()});
    Segment& cur_segment = segment_list->back();
    cur_segment.op_nodes = {logical_opnode};
    for (const DagNode* predecessor : logical_opnode->predecessors()) {
      cur_segment.predecessors.insert(
          of_dynamic_cast<const LogicalDataNode*> (predecessor));
    }
    for (const DagNode* successor : logical_opnode->successors()) {
      cur_segment.successors.insert(
          of_dynamic_cast<const LogicalDataNode*> (successor));
    }
  }
  // Init ancestors
  for (auto it = logical_dag->cbegin(); it != logical_dag->cend(); ++it) {
    // Get correct ptr
    if (typeid(*it) != typeid(LogicalOpNode)) {
      continue;
    }
    auto cur_op_node = of_dynamic_cast<const LogicalOpNode*> (&(*it));
    std::list<Segment>::iterator cur_segment = node2segment->at(cur_op_node);
    cur_segment->ancestors.clear();
    // each op predecessor
    for (const LogicalOpNode* op_predecessor : cur_op_node->op_predecessors()) {
      std::list<Segment>::iterator pre_segment = node2segment->at(op_predecessor);
      cur_segment->ancestors.insert(pre_segment->ancestors.begin(),
                                    pre_segment->ancestors.end());
      cur_segment->ancestors.insert(op_predecessor);
    }
  }
  // Init descendants
  for (auto it = logical_dag->crbegin(); it != logical_dag->crend(); ++it) {
    if (typeid(*it) != typeid(LogicalOpNode)) {
      continue;
    }
    auto cur_op_node = of_dynamic_cast<const LogicalOpNode*> (&(*it));
    std::list<Segment>::iterator cur_segment = node2segment->at(cur_op_node);
    cur_segment->descendants.clear();
    // each op successors
    for (const LogicalOpNode* op_successor : cur_op_node->op_successors()) {
      std::list<Segment>::iterator next_segment = node2segment->at(op_successor);
      cur_segment->descendants.insert(next_segment->descendants.begin(),
                                      next_segment->descendants.end());
      cur_segment->descendants.insert(op_successor);
    }
  }
}

static void ModelMergeSegments(
    std::list<Segment>* segment_list,
    std::unordered_map<const LogicalOpNode*,
                       std::list<Segment>::iterator>* node2segment) {
  for (auto& pair : *node2segment) {
    // Get cur_op_node, pre_op_node
    const LogicalOpNode* cur_op_node = pair.first;
    if (cur_op_node->layer_desc().IsElemWise() == false) {
      continue;
    }
    if (cur_op_node->parallel_conf().policy() != ParallelConf::ModelParallel) {
      continue;
    }
    CHECK_EQ(cur_op_node->op_predecessors().size(), 1);
    CHECK_EQ(cur_op_node->predecessors().size(), 1);
    const LogicalOpNode* pre_op_node = *(cur_op_node->op_predecessors().begin());
    if (pre_op_node->parallel_conf() != cur_op_node->parallel_conf()) {
      continue;
    }
    // Get segment
    std::list<Segment>::iterator pre_segment = node2segment->at(pre_op_node);
    std::list<Segment>::iterator cur_segment = pair.second;
    // Merge
    pre_segment->op_nodes.insert(pre_segment->op_nodes.end(),
                                 cur_segment->op_nodes.begin(),
                                 cur_segment->op_nodes.end());
    pre_segment->successors.insert(cur_segment->successors.begin(),
                                   cur_segment->successors.end());
    auto to_be_erased = of_dynamic_cast<LogicalDataNode*> (*(cur_op_node->predecessors().begin()));
    pre_segment->successors.erase(to_be_erased);
    for (const LogicalOpNode* node : cur_segment->op_nodes) {
      pre_segment->descendants.erase(node);
      node2segment->at(node) = pre_segment;
    }
    segment_list->erase(cur_segment);
  }
}

// from lhs to rhs
bool TryMergeWithConnect(std::list<Segment>::iterator lhs,
                         std::list<Segment>::iterator rhs) {
  // test if it can be merged
  std::unordered_set<const LogicalOpNode*> lhs_ancestors_and_lhs(lhs->ancestors);
  lhs_ancestors_and_lhs.insert(lhs->op_nodes.begin(), lhs->op_nodes.end());
  if (lhs_ancestors_and_lhs != rhs->ancestors) {
    return false;
  }
  std::unordered_set<const LogicalOpNode*> rhs_descendants_and_rhs(rhs->descendants);
  rhs_descendants_and_rhs.insert(rhs->op_nodes().begin(), rhs->op_nodes().end());
  if (rhs_descendants_and_rhs != lhs->descendants) {
    return false;
  }
  // TODO: Merge

  return true;
}

bool TryMergeWithoutConnect() {
  // test if it can be merged
  if (lhs.ancestors != rhs.ancestors || lhs.descendants != rhs.descendants) {
    return false;
  }
  // TODO: Merge
  return true;
}

// it is ugly, we can optimize it later
static void DataMergeSegments(std::list<Segment>* segment_list) {
  while (true) {
    bool has_merge = false;
    for (auto lhs = segment_list.begin(); lhs != segment_list.end(); ++lhs) {
      for (auto rhs = segment_list.begin(); rhs != segment_list.end(); ++rhs) {
        has_merge = (has_merge || TryMergeWithConnect(lhs, rhs));
        if (has_merge) {
          break;
        }
        has_merge = (has_merge || TryMergeWithConnect(rhs, lhs));
        if (has_merge) {
          break;
        }
        has_merge = (has_merge || TryMergeWithoutConnect(lhs, rhs));
        if (has_merge) {
          break;
        }
      }
      if (has_merge) {
        break;
      }
    }
    if (has_merge == false) {
      break;
    }
  }
}

void SegmentDag::Init(const std::string& dag_name,
                      std::shared_ptr<const LogicalDag> logical_dag) {
  std::list<Segment> segment_list;
  std::unordered_map<const LogicalOpNode*,
                     std::list<Segment>::iterator> node2segment;
  InitSegments(logical_dag.get(), &segment_list, &node2segment);
  ModelMergeSegments(&segment_list, &node2segment);
  node2segment.clear();
  DataMergeSegments(&segment_list);
}

} // namespace oneflow
