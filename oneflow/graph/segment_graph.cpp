#include "graph/segment_graph.h"
#include "glog/logging.h"
#include <list>

namespace oneflow {

namespace {

struct Segment {
  // nodes belong to this Segment
  std::vector<const LogicalNode*> nodes;
  // ancestors, descendants of nodes
  std::unordered_set<const LogicalNode*> ancestors;
  std::unordered_set<const LogicalNode*> descendants;
  // ancestors_and_this = nodes + ancestors
  // descendants_and_this = nodes + descendants
  std::unordered_set<const LogicalNode*> ancestors_and_this;
  std::unordered_set<const LogicalNode*> descendants_and_this;
};

using SegmentIt = std::list<Segment>::iterator;

void SetSegmentNodeWithSegmentIt(SegmentNode* seg_node,
                                 SegmentIt seg_it) {
  CHECK_EQ(seg_it->nodes.empty(), false);
  seg_node->mutable_parallel_desc_ptr() =
      seg_it->nodes.front()->parallel_desc_ptr();
  for (const LogicalNode* logical_node : seg_it->nodes) {
    seg_node->mutable_op_vec().push_back(
        logical_node->op_ptr());
  }
}

void InitSegments(
    const LogicalGraph* logical_graph,
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalNode*,
                       SegmentIt>* logical_node2seg_it) {
  // Init one Segment with one Node
  seg_list->clear();
  logical_node2seg_it->clear();
  // Init ops
  for (const std::unique_ptr<Node>& node : logical_graph->node_vec()) {
    auto logical_node = of_dynamic_cast<const LogicalNode*>(node.get());
    seg_list->emplace_back();
    logical_node2seg_it->insert({logical_node, --seg_list->end()});
    Segment& cur_segment = seg_list->back();
    cur_segment.nodes = {logical_node};
  }
  // Init ancestors
  for (auto it = logical_graph->cbegin(); it != logical_graph->cend(); ++it) {
    // Get correct ptr
    auto cur_node = of_dynamic_cast<const LogicalNode*> (&(*it));
    SegmentIt cur_segment = logical_node2seg_it->at(cur_node);
    cur_segment->ancestors.clear();
    // each predecessor
    for (const Edge* edge : cur_node->in_edges()) {
      auto logi_pre = of_dynamic_cast<const LogicalNode*> (edge->src_node());
      SegmentIt pre_segment = logical_node2seg_it->at(logi_pre);
      // ancestors
      cur_segment->ancestors.insert(pre_segment->ancestors.begin(),
                                    pre_segment->ancestors.end());
      cur_segment->ancestors.insert(logi_pre);
      // ancestors_and_this
      cur_segment->ancestors_and_this = cur_segment->ancestors;
      cur_segment->ancestors_and_this.insert(cur_segment->nodes.begin(),
                                             cur_segment->nodes.end());
    }
  }
  // Init descendants
  for (auto it = logical_graph->crbegin(); it != logical_graph->crend(); ++it) {
    auto cur_node = of_dynamic_cast<const LogicalNode*> (&(*it));
    SegmentIt cur_segment = logical_node2seg_it->at(cur_node);
    cur_segment->descendants.clear();
    // each successors
    for (const Edge* edge : cur_node->out_edges()) {
      auto logi_succ = of_dynamic_cast<const LogicalNode*> (edge->dst_node());
      SegmentIt next_segment = logical_node2seg_it->at(logi_succ);
      // descendants
      cur_segment->descendants.insert(next_segment->descendants.begin(),
                                      next_segment->descendants.end());
      cur_segment->descendants.insert(logi_succ);
      // descendants_and_this
      cur_segment->descendants_and_this = cur_segment->descendants;
      cur_segment->descendants_and_this.insert(cur_segment->nodes.begin(),
                                               cur_segment->nodes.end());
    }
  }
}

void ModelMergeSegments(
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalNode*,
                       SegmentIt>* logical_node2seg_it) {
  for (auto& pair : *logical_node2seg_it) {
    // Get cur_node, pre_node
    const LogicalNode* cur_node = pair.first;
    if (cur_node->op().IsElemWise() == false) {
      continue;
    }
    if (cur_node->parallel_desc().policy() != ParallelDesc::kModelParallel) {
      continue;
    }
    CHECK_EQ(cur_node->in_edges().size(), 1);
    CHECK_EQ(cur_node->in_edges().size(), 1);
    auto pre_node =
        of_dynamic_cast<const LogicalNode*>
            ((*(cur_node->in_edges().begin()))->src_node());
    if (pre_node->parallel_desc() != cur_node->parallel_desc()) {
      continue;
    }
    // Get segment
    SegmentIt pre_segment = logical_node2seg_it->at(pre_node);
    SegmentIt cur_segment = pair.second;
    // Merge
    pre_segment->nodes.insert(pre_segment->nodes.end(),
                                 cur_segment->nodes.begin(),
                                 cur_segment->nodes.end());
    for (const LogicalNode* node : cur_segment->nodes) {
      pre_segment->descendants.erase(node);
      logical_node2seg_it->at(node) = pre_segment;
    }
    seg_list->erase(cur_segment);
  }
}

bool TryMergeWithConnect(
    const LogicalNode* up_node,
    const LogicalNode* bottom_node,
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalNode*,
                       SegmentIt>* logical_node2seg_it) {
  // Get segment
  SegmentIt up_seg = logical_node2seg_it->at(up_node);
  SegmentIt bottom_seg = logical_node2seg_it->at(bottom_node);
  // if it can be merged
  if (up_seg->ancestors_and_this != bottom_seg->ancestors
      || bottom_seg->descendants_and_this != up_seg->descendants) {
    return false;
  }
  // Merge
  if (up_seg->nodes.size() > bottom_seg->nodes.size()) {
    for (const LogicalNode* node : bottom_seg->nodes) {
      up_seg->nodes.push_back(node);
      up_seg->descendants.erase(node);
      logical_node2seg_it->at(node) = up_seg;
    }
    seg_list->erase(bottom_seg);
  } else {
    for (const LogicalNode* node : up_seg->nodes) {
      bottom_seg->nodes.push_back(node);
      bottom_seg->ancestors.erase(node);
      logical_node2seg_it->at(node) = bottom_seg;
    }
    seg_list->erase(up_seg);
  }
  return true;
}

bool TryMergeWithoutConnect(
    const LogicalNode* lhs_node,
    const LogicalNode* rhs_node,
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalNode*,
                       SegmentIt>* logical_node2seg_it) {
  // Get segment
  SegmentIt lhs_segment = logical_node2seg_it->at(lhs_node);
  SegmentIt rhs_segment = logical_node2seg_it->at(rhs_node);
  // if it can be merged
  if (lhs_segment->ancestors != rhs_segment->ancestors
      || lhs_segment->descendants != rhs_segment->descendants) {
    return false;
  }
  // Merge
  // If this is bottleneck, we can optimze it by compare the size of lhs,rhs
  for (const LogicalNode* node : rhs_segment->nodes) {
    lhs_segment->nodes.push_back(node);
    lhs_segment->ancestors_and_this.insert(node);
    lhs_segment->descendants_and_this.insert(node);
    logical_node2seg_it->at(node) = lhs_segment;
  }
  seg_list->erase(rhs_segment);
  return true;
}

void Traverse(const LogicalNode* seed_node,
              const std::vector<const LogicalNode*>& data_parallel_node,
              std::list<Segment>* seg_list,
              std::unordered_map<const LogicalNode*, bool>* done,
              std::unordered_map<const LogicalNode*,
                                 SegmentIt>* logical_node2seg_it) {
  done->at(seed_node) = true;
  while (true) {
    bool has_merged = false;
    for (const LogicalNode* node : data_parallel_node) {
      if (done->at(node)) { continue; }
      if (seed_node->parallel_desc() != node->parallel_desc()) {
        continue;
      }
      if (TryMergeWithConnect(seed_node, node, seg_list, logical_node2seg_it)
          || TryMergeWithConnect(node, seed_node, seg_list, logical_node2seg_it)
          || TryMergeWithoutConnect(seed_node, node, seg_list, logical_node2seg_it)) {
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
    const LogicalGraph* logical_graph,
    std::list<Segment>* seg_list,
    std::unordered_map<const LogicalNode*,
                       SegmentIt>* logical_node2seg_it) {
  std::vector<const LogicalNode*> data_parallel_node;
  std::unordered_map<const LogicalNode*, bool> done;
  for (const auto& pair : *logical_node2seg_it) {
    if (pair.first->parallel_desc().policy() == ParallelDesc::kDataParallel
        && logical_graph->IsFirstNode(pair.first) == false) {
      data_parallel_node.push_back(pair.first);
      done[pair.first] = false;
    }
  }
  for (const LogicalNode* seed_node : data_parallel_node) {
    if (done.at(seed_node) == false) {
      Traverse(seed_node,
               data_parallel_node,
               seg_list,
               &done,
               logical_node2seg_it);
    }
  }
}

} // namespace

void SegmentGraph::Init(const LogicalGraph* logical_graph) {
  // Build Segment
  std::list<Segment> seg_list;
  std::unordered_map<const LogicalNode*,
                     SegmentIt> logical_node2seg_it;
  InitSegments(logical_graph, &seg_list, &logical_node2seg_it);
  ModelMergeSegments(&seg_list, &logical_node2seg_it);
  DataMergeSegments(logical_graph,
                    &seg_list,
                    &logical_node2seg_it);
  // Init segment_nodes
  auto seg_it_hash = [](const SegmentIt& seg_it) {
    return std::hash<Segment*> ()(&(*seg_it));
  };
  std::unordered_map<SegmentIt, SegmentNode*, decltype(seg_it_hash)>
      seg_it2seg_node(0, seg_it_hash);
  std::unordered_map<SegmentNode*,
                     std::unordered_set<SegmentNode*>> seg_node2pre;
  for (auto seg_it = seg_list.begin(); seg_it != seg_list.end(); ++seg_it) {
    SegmentNode* seg_node = NewSegmentNode();
    seg_it2seg_node[seg_it] = seg_node;
    seg_node2pre[seg_node] = {};
    SetSegmentNodeWithSegmentIt(seg_node, seg_it);
  }
  // Record the predecessor
  for (auto seg_it = seg_list.begin(); seg_it != seg_list.end(); ++seg_it) {
    SegmentNode* seg_node = seg_it2seg_node.at(seg_it);
    for (const LogicalNode* logi_node : seg_it->nodes) {
      for (auto logi_in_edge : logi_node->in_edges()) {
        auto pre_seg_it =
            logical_node2seg_it.at(
                of_dynamic_cast<const LogicalNode*>(logi_in_edge->src_node()));
        auto pre_seg_node = seg_it2seg_node.at(pre_seg_it);
        seg_node2pre.at(seg_node).insert(pre_seg_node);
      }
    }
  }
  // Connect
  for (auto& pair : seg_node2pre) {
    SegmentNode* cur_node = pair.first;
    for (SegmentNode* pre_node : pair.second) {
      Connect(pre_node, NewSegmentEdge(), cur_node);
    }
  }
  // Post processing
  UpdateStartAndStop();
}

} // namespace oneflow
