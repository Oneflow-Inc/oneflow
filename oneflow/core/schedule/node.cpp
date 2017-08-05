/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/node.h"

namespace oneflow {
namespace schedule {

void GraphNode::InitSourceAndSink() {
  mut_source() = mut_fake_node_mgr().Create("source");
  mut_sink() = mut_fake_node_mgr().Create("sink");
}

int GraphNode::LossNodes(std::list<Node*>* l) const {
  return loss_arc_mgr().Output(this, l);
}

void GraphNode::UpdateSourceAndSink() {
  std::list<Arc*> arcs;
  arc_mgr().OutputArc(source(), &arcs);
  arc_mgr().InputArc(sink(), &arcs);
  for (auto arc : arcs) { mut_arc_mgr().Delete(arc->id()); }
  children_arc_mgr().Output(this, [&](Node* leaf) {
    if (!arc_mgr().Input(leaf)) {
      mut_arc_mgr().CreateIfNotFound(source(), leaf);
    }
    if (!arc_mgr().Output(leaf)) {
      mut_arc_mgr().CreateIfNotFound(leaf, sink());
    }
  });
}

void GraphNode::ForeachArc(const std::function<void(Arc*)>& cb) const {
  arc_mgr().OutputArc(source(), cb);
  children_arc_mgr().Output(
      this, [&](Node* child) { arc_mgr().OutputArc(child, cb); });
}

void GraphNode::ForeachNodeWithSourceAndSink(
    const std::function<void(Node*)>& cb) const {
  cb(source());
  ForeachNode(cb);
  cb(sink());
}
void GraphNode::ForeachNode(const std::function<void(Node*)>& cb) const {
  cb(source());
  children_arc_mgr().Output(this, cb);
  cb(sink());
}

void GraphNode::ForeachRegstDesc(const std::function<void(Node*)>& cb) const {
  children_arc_mgr().Output(
      this, [&](Node* node) { produced_regst_desc_mgr().Output(node, cb); });
}

uint32_t GraphNode::Depth() const {
  auto depth = source()->depth();
  return depth ? depth - 1 : 0;
}

uint32_t GraphNode::DeviceCount() const {
  std::unordered_set<Node*> devices;
  children_arc_mgr().Output(this, [&](Node* node) {
    Node* device = nullptr;
    device_arc_mgr().Output(node, &device);
    devices.insert(device);
  });
  return devices.size();
}

void GraphNode::WalkArcReverse(const std::function<void(Arc*)>& cb) {
  WalkReverse([&](Node* node) {
    arc_mgr().OutputArc(node, [&](Arc* arc) { cb(arc); });
  });
}

void GraphNode::WalkReverse(const std::function<void(Node*)>& cb) {
  auto next = std::unordered_set<Node*>{sink()};
  auto marked = std::unordered_set<Node*>{};
  while (next.size()) {
    auto queue = std::list<Node*>(next.begin(), next.end());
    for (const auto& node : queue) {
      cb(node);
      marked.insert(node);
      next.erase(node);
      arc_mgr().InputArc(node, [&](Arc* arc) {
        bool all_marked = true;
        arc_mgr().Output(arc->from(), [&](Node* from) {
          if (all_marked && marked.find(from) == marked.end()) {
            all_marked = false;
          }
        });
        if (all_marked && marked.find(arc->from()) == marked.end()) {
          next.insert(arc->from());
        }
      });
    }
  }
}

void GraphNode::WalkArc(const std::function<void(Arc*)>& cb) {
  Walk([&](Node* node) { arc_mgr().InputArc(node, cb); });
}

void GraphNode::Walk(const std::function<void(Node*)>& cb) {
  auto next = std::unordered_set<Node*>{source()};
  auto marked = std::unordered_set<Node*>{};
  while (next.size()) {
    auto queue = std::list<Node*>(next.begin(), next.end());
    for (const auto& node : queue) {
      cb(node);
      marked.insert(node);
      next.erase(node);
      arc_mgr().OutputArc(node, [&](Arc* arc) {
        bool all_marked = true;
        arc_mgr().Input(arc->to(), [&](Node* from) {
          if (all_marked && marked.find(from) == marked.end()) {
            all_marked = false;
          }
        });
        if (all_marked && marked.find(arc->to()) == marked.end()) {
          next.insert(arc->to());
        }
      });
    }
  }
}

void GraphNode::InitAscendentArc() {
  Walk([&](Node* node) {
    arc_mgr().Input(node, [&](Node* prev) {
      std::list<Node*> l;
      ascendent_arc_mgr().Output(prev, &l);
      for (Node* asc : l) {
        mut_ascendent_arc_mgr().CreateIfNotFound(node, asc);
      }
      mut_ascendent_arc_mgr().CreateIfNotFound(node, prev);
    });
  });
}

void GraphNode::ForeachAscendent(Node* node,
                                 const std::function<void(Node*)>& cb) const {
  ascendent_arc_mgr().Output(node, cb);
}

void GraphNode::ForeachDescendent(Node* node,
                                  const std::function<void(Node*)>& cb) const {
  ascendent_arc_mgr().Input(node, cb);
}

void GraphNode::InitDepth() {
  WalkReverse([&](Node* node) {
    int depth = -1;
    arc_mgr().Output(node,
                     [&](Node* to) { depth = std::max(depth, to->depth()); });
    node->mut_depth() = depth + 1;
  });
}

}  // namespace schedule
}  // namespace oneflow
