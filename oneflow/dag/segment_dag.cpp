#include "dag/segment_dag.h"
#include "glog/logging.h"
#include <list>

namespace oneflow {

namespace {

struct Segment {
  // nodes belong to this Segment
  std::vector<const LogicalOpNode*> op_nodes;
  // ancestors, descendants of op_nodes
  std::unordered_set<const LogicalOpNode*> ancestors;
  std::unordered_set<const LogicalOpNode*> descendants;
  // ancestors_and_this = op_nodes + ancestors
  // descendants_and_this = op_nodes + descendants
  std::unordered_set<const LogicalOpNode*> ancestors_and_this;
  std::unordered_set<const LogicalOpNode*> descendants_and_this;
};

using SegmentIt = std::list<Segment>::iterator;

void SetSegmentOpNodeWithSegmentIt(SegmentOpNode* seg_opnode,
                                   SegmentIt seg_it) {
  CHECK_EQ(seg_it->op_nodes.empty(), false);
  seg_opnode->mutable_parallel_desc_ptr() =
      seg_it->op_nodes.front()->parallel_desc_ptr();
  for (const LogicalOpNode* logical_opnode : seg_it->op_nodes) {
    seg_opnode->mutable_layer_desc_vec().push_back(
        logical_opnode->layer_desc_ptr());
  }
}

void InitSegments(
    const LogicalDag* logical_dag,
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalOpNode*,
                       SegmentIt>* logical_opnode2seg_it) {
  // Init one Segment with one OpNode
  seg_list->clear();
  logical_opnode2seg_it->clear();
  // Init layers
  for (const OpNode* op_node : logical_dag->op_node_vec()) {
    auto logical_opnode = of_dynamic_cast<const LogicalOpNode*>(op_node);
    seg_list->emplace_back();
    logical_opnode2seg_it->insert({logical_opnode, --seg_list->end()});
    Segment& cur_segment = seg_list->back();
    cur_segment.op_nodes = {logical_opnode};
  }
  // Init ancestors
  for (auto it = logical_dag->cbegin(); it != logical_dag->cend(); ++it) {
    // Get correct ptr
    if (typeid(*it) != typeid(LogicalOpNode)) {
      continue;
    }
    auto cur_op_node = of_dynamic_cast<const LogicalOpNode*> (&(*it));
    SegmentIt cur_segment = logical_opnode2seg_it->at(cur_op_node);
    cur_segment->ancestors.clear();
    // each op predecessor
    for (const OpNode* op_pre : cur_op_node->op_predecessors()) {
      auto logi_op_pre = of_dynamic_cast<const LogicalOpNode*> (op_pre);
      SegmentIt pre_segment = logical_opnode2seg_it->at(logi_op_pre);
      // ancestors
      cur_segment->ancestors.insert(pre_segment->ancestors.begin(),
                                    pre_segment->ancestors.end());
      cur_segment->ancestors.insert(logi_op_pre);
      // ancestors_and_this
      cur_segment->ancestors_and_this = cur_segment->ancestors;
      cur_segment->ancestors_and_this.insert(cur_segment->op_nodes.begin(),
                                             cur_segment->op_nodes.end());
    }
  }
  // Init descendants
  for (auto it = logical_dag->crbegin(); it != logical_dag->crend(); ++it) {
    if (typeid(*it) != typeid(LogicalOpNode)) {
      continue;
    }
    auto cur_op_node = of_dynamic_cast<const LogicalOpNode*> (&(*it));
    SegmentIt cur_segment = logical_opnode2seg_it->at(cur_op_node);
    cur_segment->descendants.clear();
    // each op successors
    for (const OpNode* op_succ : cur_op_node->op_successors()) {
      auto logi_op_succ = of_dynamic_cast<const LogicalOpNode*> (op_succ);
      SegmentIt next_segment = logical_opnode2seg_it->at(logi_op_succ);
      // descendants
      cur_segment->descendants.insert(next_segment->descendants.begin(),
                                      next_segment->descendants.end());
      cur_segment->descendants.insert(logi_op_succ);
      // descendants_and_this
      cur_segment->descendants_and_this = cur_segment->descendants;
      cur_segment->descendants_and_this.insert(cur_segment->op_nodes.begin(),
                                               cur_segment->op_nodes.end());
    }
  }
}

void ModelMergeSegments(
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalOpNode*,
                       SegmentIt>* logical_opnode2seg_it) {
  for (auto& pair : *logical_opnode2seg_it) {
    // Get cur_op_node, pre_op_node
    const LogicalOpNode* cur_op_node = pair.first;
    if (cur_op_node->layer_desc().IsElemWise() == false) {
      continue;
    }
    if (cur_op_node->parallel_desc().policy() != ParallelDesc::kModelParallel) {
      continue;
    }
    CHECK_EQ(cur_op_node->op_predecessors().size(), 1);
    CHECK_EQ(cur_op_node->predecessors().size(), 1);
    auto pre_op_node =
        of_dynamic_cast<const LogicalOpNode*>(*(cur_op_node->op_predecessors().begin()));
    if (pre_op_node->parallel_desc() != cur_op_node->parallel_desc()) {
      continue;
    }
    // Get segment
    SegmentIt pre_segment = logical_opnode2seg_it->at(pre_op_node);
    SegmentIt cur_segment = pair.second;
    // Merge
    pre_segment->op_nodes.insert(pre_segment->op_nodes.end(),
                                 cur_segment->op_nodes.begin(),
                                 cur_segment->op_nodes.end());
    for (const LogicalOpNode* node : cur_segment->op_nodes) {
      pre_segment->descendants.erase(node);
      logical_opnode2seg_it->at(node) = pre_segment;
    }
    seg_list->erase(cur_segment);
  }
}

bool TryMergeWithConnect(
    const LogicalOpNode* up_node,
    const LogicalOpNode* bottom_node,
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalOpNode*,
                       SegmentIt>* logical_opnode2seg_it) {
  // Get segment
  SegmentIt up_seg = logical_opnode2seg_it->at(up_node);
  SegmentIt bottom_seg = logical_opnode2seg_it->at(bottom_node);
  // if it can be merged
  if (up_seg->ancestors_and_this != bottom_seg->ancestors
      || bottom_seg->descendants_and_this != up_seg->descendants) {
    return false;
  }
  // Merge
  if (up_seg->op_nodes.size() > bottom_seg->op_nodes.size()) {
    for (const LogicalOpNode* node : bottom_seg->op_nodes) {
      up_seg->op_nodes.push_back(node);
      up_seg->descendants.erase(node);
      logical_opnode2seg_it->at(node) = up_seg;
    }
    seg_list->erase(bottom_seg);
  } else {
    for (const LogicalOpNode* node : up_seg->op_nodes) {
      bottom_seg->op_nodes.push_back(node);
      bottom_seg->ancestors.erase(node);
      logical_opnode2seg_it->at(node) = bottom_seg;
    }
    seg_list->erase(up_seg);
  }
  return true;
}

bool TryMergeWithoutConnect(
    const LogicalOpNode* lhs_node,
    const LogicalOpNode* rhs_node,
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalOpNode*,
                       SegmentIt>* logical_opnode2seg_it) {
  // Get segment
  SegmentIt lhs_segment = logical_opnode2seg_it->at(lhs_node);
  SegmentIt rhs_segment = logical_opnode2seg_it->at(rhs_node);
  // if it can be merged
  if (lhs_segment->ancestors != rhs_segment->ancestors
      || lhs_segment->descendants != rhs_segment->descendants) {
    return false;
  }
  // Merge
  // If this is bottleneck, we can optimze it by compare the size of lhs,rhs
  for (const LogicalOpNode* node : rhs_segment->op_nodes) {
    lhs_segment->op_nodes.push_back(node);
    lhs_segment->ancestors_and_this.insert(node);
    lhs_segment->descendants_and_this.insert(node);
    logical_opnode2seg_it->at(node) = lhs_segment;
  }
  seg_list->erase(rhs_segment);
  return true;
}

void Traverse(const LogicalOpNode* seed_node,
              const std::vector<const LogicalOpNode*>& data_parallel_node,
              std::list<Segment>* seg_list,
              std::unordered_map<const LogicalOpNode*, bool>* done,
              std::unordered_map<const LogicalOpNode*,
                                 SegmentIt>* logical_opnode2seg_it) {
  done->at(seed_node) = true;
  while (true) {
    bool has_merged = false;
    for (const LogicalOpNode* node : data_parallel_node) {
      if (done->at(node)) { continue; }
      if (seed_node->parallel_desc() != node->parallel_desc()) {
        continue;
      }
      if (TryMergeWithConnect(seed_node, node, seg_list, logical_opnode2seg_it)
          || TryMergeWithConnect(node, seed_node, seg_list, logical_opnode2seg_it)
          || TryMergeWithoutConnect(seed_node, node, seg_list, logical_opnode2seg_it)) {
        done->at(node) = true;
        has_merged = true;
        break;
      }
    }
    if (has_merged == false) {
      break;
    }
  }
}

void DataMergeSegments(
    const LogicalDag* logical_dag,
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalOpNode*,
                       SegmentIt>* logical_opnode2seg_it) {
  std::vector<const LogicalOpNode*> data_parallel_node;
  std::unordered_map<const LogicalOpNode*, bool> done;
  for (const auto& pair : *logical_opnode2seg_it) {
    if (pair.first->parallel_desc().policy() == ParallelDesc::kDataParallel
        && logical_dag->IsFirstNode(pair.first) == false) {
      data_parallel_node.push_back(pair.first);
      done[pair.first] = false;
    }
  }
  for (const LogicalOpNode* seed_node : data_parallel_node) {
    if (done.at(seed_node) == false) {
      Traverse(seed_node,
               data_parallel_node,
               seg_list,
               &done,
               logical_opnode2seg_it);
    }
  }
}

} // namespace

void SegmentDag::Init(const std::string& dag_name,
                      std::shared_ptr<const LogicalDag> logical_dag) {
  // Build Segment
  std::list<Segment> seg_list;
  std::unordered_map<const LogicalOpNode*,
                     SegmentIt> logical_opnode2seg_it;
  InitSegments(logical_dag.get(), &seg_list, &logical_opnode2seg_it);
  ModelMergeSegments(&seg_list, &logical_opnode2seg_it);
  DataMergeSegments(logical_dag.get(),
                    &seg_list,
                    &logical_opnode2seg_it);
  // Init segment_op_nodes
  auto seg_it_hash = [](const SegmentIt& seg_it) {
    return std::hash<Segment*> ()(&(*seg_it));
  };
  std::unordered_map<SegmentIt, SegmentOpNode*, decltype(seg_it_hash)>
      seg_it2seg_opnode(0, seg_it_hash);
  std::unordered_map<SegmentOpNode*,
                     std::unordered_set<SegmentOpNode*>> seg_op_node2pre;
  for (auto seg_it = seg_list.begin(); seg_it != seg_list.end(); ++seg_it) {
    SegmentOpNode* seg_opnode = NewSegmentOpNode();
    seg_it2seg_opnode[seg_it] = seg_opnode;
    seg_op_node2pre[seg_opnode] = {};
    SetSegmentOpNodeWithSegmentIt(seg_opnode, seg_it);
  }
  // Record the predecessor
  for (auto seg_it = seg_list.begin(); seg_it != seg_list.end(); ++seg_it) {
    SegmentOpNode* seg_opnode = seg_it2seg_opnode.at(seg_it);
    for (const LogicalOpNode* logi_opnode : seg_it->op_nodes) {
      for (auto pre_logi_opnode : logi_opnode->op_predecessors()) {
        auto pre_seg_it = logical_opnode2seg_it.at(of_dynamic_cast<const LogicalOpNode*>(pre_logi_opnode));
        auto pre_seg_opnode = seg_it2seg_opnode.at(pre_seg_it);
        seg_op_node2pre.at(seg_opnode).insert(pre_seg_opnode);
      }
    }
  }
  // Connect
  for (auto& pair : seg_op_node2pre) {
    SegmentOpNode* cur_node = pair.first;
    for (SegmentOpNode* pre_node : pair.second) {
      SegmentDataNode* data_node_ptr = NewSegmentDataNode();
      cur_node->AddPredecessor(data_node_ptr);
      data_node_ptr->AddPredecessor(pre_node);
    }
  }
  // Post processing
  ConnectStartAndStop();
  ConnectOpNodeExtraPtr();
}

} // namespace oneflow
