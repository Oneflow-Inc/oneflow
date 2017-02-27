#include "dag/segment_dag.h"
#include "glog/logging.h"
#include <list>

namespace oneflow {

struct Segment {
  // Main
  std::vector<const LogicalOpNode*> op_nodes;
  // Attached Properties
  std::unordered_set<const LogicalOpNode*> ancestors;
  std::unordered_set<const LogicalOpNode*> descendants;
};

static void InitSegments(
    const LogicalDag* logical_dag,
    std::list<Segment>* segment_list,
    std::unordered_map<const LogicalOpNode*,
                       std::list<Segment>::iterator>* opnode2segment) {
  // Init one Segment with one OpNode
  segment_list->clear();
  opnode2segment->clear();
  // Init layers
  for (const std::unique_ptr<OpNode>& op_node : logical_dag->op_node_vec()) {
    auto logical_opnode = of_dynamic_cast<const LogicalOpNode*>(op_node.get());
    segment_list->emplace_back();
    opnode2segment->insert({logical_opnode, --segment_list->end()});
    Segment& cur_segment = segment_list->back();
    cur_segment.op_nodes = {logical_opnode};
  }
  // Init ancestors
  for (auto it = logical_dag->cbegin(); it != logical_dag->cend(); ++it) {
    // Get correct ptr
    if (typeid(*it) != typeid(LogicalOpNode)) {
      continue;
    }
    auto cur_op_node = of_dynamic_cast<const LogicalOpNode*> (&(*it));
    std::list<Segment>::iterator cur_segment = opnode2segment->at(cur_op_node);
    cur_segment->ancestors.clear();
    // each op predecessor
    for (const LogicalOpNode* op_predecessor : cur_op_node->op_predecessors()) {
      std::list<Segment>::iterator pre_segment = opnode2segment->at(op_predecessor);
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
    std::list<Segment>::iterator cur_segment = opnode2segment->at(cur_op_node);
    cur_segment->descendants.clear();
    // each op successors
    for (const LogicalOpNode* op_successor : cur_op_node->op_successors()) {
      std::list<Segment>::iterator next_segment = opnode2segment->at(op_successor);
      cur_segment->descendants.insert(next_segment->descendants.begin(),
                                      next_segment->descendants.end());
      cur_segment->descendants.insert(op_successor);
    }
  }
}

static void ModelMergeSegments(
    std::list<Segment>* segment_list,
    std::unordered_map<const LogicalOpNode*,
                       std::list<Segment>::iterator>* opnode2segment) {
  for (auto& pair : *opnode2segment) {
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
    std::list<Segment>::iterator pre_segment = opnode2segment->at(pre_op_node);
    std::list<Segment>::iterator cur_segment = pair.second;
    // Merge
    pre_segment->op_nodes.insert(pre_segment->op_nodes.end(),
                                 cur_segment->op_nodes.begin(),
                                 cur_segment->op_nodes.end());
    for (const LogicalOpNode* node : cur_segment->op_nodes) {
      pre_segment->descendants.erase(node);
      opnode2segment->at(node) = pre_segment;
    }
    segment_list->erase(cur_segment);
  }
}

bool TryMergeWithConnect(const LogicalOpNode* up_node,
                         const LogicalOpNode* bottom_node,
                         std::list<Segment>* segment_list,
                         std::unordered_map<const LogicalOpNode*
                                            std::list<Segment>::iterator>* opnode2segment) {
  std::list<Segment>::iterator up_segment = opnode2segment.at(up_node);
  std::list<Segment>::iterator bottom_segment = opnode2segment.at(bottom_node);
  // if it can be merged
  std::unordered_set<const LogicalOpNode*> up_ancestors_and_up(up_segment->ancestors);
  up_ancestors_and_up.insert(up_segment->op_nodes.begin(), up_segment->op_nodes.end());
  if (up_ancestors_and_up != rhs->ancestors) {
    return false;
  }
  std::unordered_set<const LogicalOpNode*> bottom_descendants_and_bottom(bottom_segment->descendants);
  bottom_descendants_and_bottom.insert(bottom_segment->op_nodes().begin(), bottom_segment->op_nodes().end());
  if (bottom_descendants_and_bottom != up_segment->descendants) {
    return false;
  }
  // Merge
  up_segment->op_nodes.insert(up_segment->op_nodes.end(),
                              bottom_segment->op_nodes.begin(),
                              bottom_segment->op_nodes.end());
  for (const LogicalOpNode* node : bottom_segment->op_nodes()) {
    up_segment->descendants.erase(node);
    opnode2segment.at(node) = up_segment;
  }
  segment_list.erase(bottom_segment);
  return true;
}

bool TryMergeWithoutConnect(const LogicalOpNode* lhs_node,
                            const LogicalOpNode* rhs_node,
                            std::list<Segment>* segment_list,
                            std::unordered_map<const LogicalOpNode*
                                               std::list<Segment>::iterator>* opnode2segment) {
  std::list<Segment>::iterator lhs_segment = opnode2segment.at(lhs_node);
  std::list<Segment>::iterator rhs_segment = opnode2segment.at(rhs_node);
  // if it can be merged
  if (lhs_segment->ancestors != rhs_segment->ancestors
      || lhs_segment->descendants != rhs_segment->descendants) {
    return false;
  }
  // Merge
  lhs_segment->op_nodes.insert(lhs_segment->op_nodes.end(),
                               rhs_segment->op_nodes.begin(),
                               rhs_segment->op_nodes.end());
  for (const LogicalOpNode* node : rhs_segment->op_nodes()) {
    opnode2segment.at(node) = lhs_segment;
  }
  segment_list.erase(rhs_segment);
  return true;
}

static void Traverse(const LogicalOpNode* seed_node,
                     const std::vector<const LogicalOpNode*>& data_parallel_node,
                     std::list<Segment>* segment_list,
                     std::unordered_map<const LogicalOpNode*, bool>* done,
                     std::unordered_map<const LogicalOpNode*,
                                        std::list<Segment>::iterator>* opnode2segment) {
  done[seed_node] = true;
  while (true) {
    bool has_merged = false;
    for (const LogicalOpNode* node : data_parallel_node) {
      if (done[node]) { continue; }
      if (seed_node->parallel_conf() != node->parallel_conf()) {
        continue;
      }
      if (TryMergeWithConnect(seed_node, node)
          || TryMergeWithConnect(node, seed_node)
          || TryMergeWithoutConnect(seed_node, node)) {
        done.at(node) = true;
        has_merged = true;
        break;
      }
    }
    if (has_merged == false) {
      break;
    }
  }
}

static void DataMergeSegments(
    const LogicalDag* logical_dag,
    std::list<Segment>* segment_list,
    std::unordered_map<const LogicalOpNode*,
                       std::list<Segment>::iterator>* opnode2segment) {
  std::vector<const LogicalOpNode*> data_parallel_node;
  std::unordered_map<const LogicalOpNode*, bool> done;
  for (const auto& pair : *opnode2segment) {
    if (pair.first->parallel_conf().policy() == ParallelConf::DataParallel
        && logical_dag.IsFirstNode(pair.first) == false) {
      data_parallel_node.push_back(pair.first);
      done[pair.first] = false;
    }
  }
  for (const LogicalOpNode* seed_node : data_parallel_node) {
    if (done[seed_node] == false) {
      Traverse(seed_node, data_parallel_node, &segment_list, &done, &opnode2segment);
    }
  }
}

void SegmentDag::Init(const std::string& dag_name,
                      std::shared_ptr<const LogicalDag> logical_dag) {
  std::list<Segment> segment_list;
  std::unordered_map<const LogicalOpNode*,
                     std::list<Segment>::iterator> opnode2segment;
  InitSegments(logical_dag.get(), &segment_list, &opnode2segment);
  ModelMergeSegments(&segment_list, &opnode2segment);
  DataMergeSegments(logical_dag.get(), &segment_list, &opnode2segment);
  // Build Dag
  // TODO: we should care the first node
  std::unordered_map<std::list<Segment>::iterator, SegmentOpNode*> iter2segment_opnode;
  std::unordered_map<SegmentOpNode*, std::unordered_set<SegmentOpNode*>> segment_op_node2pre;
  for (auto iter = segment_list.begin(); iter != segment_list.end(); ++iter) {
    SegmentOpNode* new_node = NewSegmentOpNode();
    iter2segment_opnode[iter] = new_node;
    segment_op_node2pre[new_node] = {};
  }
  for (auto cur_segment = segment_list.begin(); cur_segment != segment_list.end(); ++cur_segment) {
    SegmentOpNode* cur_segment_op_node = iter2segment_opnode.at(cur_segment);
    for (const LogicalOpNode* cur_logical_op_node : cur_segment->op_nodes) {
      for (const LogicalOpNode* pre_logical_op_node : cur_logical_op_node->op_predecessors()) {
        SegmentOpNode* pre_segment_op_node = iter2segment_opnode.at(opnode2segment.at(pre_logical_op_node));
        segment_op_node2pre.at(cur_segment_op_node).insert(pre_segment_op_node);
      }
    }
  }
  for (auto& pair : segment_op_node2pre) {
    SegmentOpNode* cur_node = pair.first;
    for (SegmentOpNode* pre_node : pair.second) {
      SegmentDataNode* data_node_ptr = NewSegmentDataNode();
      cur_node->AddPredecessor(data_node_ptr);
      data_node_ptr->AddPredecessor(pre_node);
    }
  }
  ConnectStartAndStop();
}

} // namespace oneflow
