/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/node.h"

namespace oneflow {
namespace schedule {

void SGraph::InitSourceAndSink() {
  mut_source() = mut_fake_node_mgr().Create("source");
  mut_sink() = mut_fake_node_mgr().Create("sink");
}

int SGraph::LossNodes(std::list<Node*>* l) const {
  return loss_arc_mgr().Output(this, l);
}

void SGraph::UpdateSourceAndSink() {
  std::list<Arc<Node>*> arcs;
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

void SGraph::ForeachArc(const std::function<void(Arc<Node>*)>& cb) const {
  arc_mgr().OutputArc(source(), cb);
  children_arc_mgr().Output(
      this, [&](Node* child) { arc_mgr().OutputArc(child, cb); });
}

void SGraph::ForeachNodeWithSourceAndSink(
    const std::function<void(Node*)>& cb) const {
  cb(source());
  ForeachNode(cb);
  cb(sink());
}
void SGraph::ForeachNode(const std::function<void(Node*)>& cb) const {
  cb(source());
  children_arc_mgr().Output(this, cb);
  cb(sink());
}

void SGraph::ForeachRegstDesc(const std::function<void(RegstDesc*)>& cb) const {
  children_arc_mgr().Output(
      this, [&](Node* node) { produced_regst_desc_mgr().Output(node, cb); });
}

uint32_t SGraph::Depth() const {
  auto depth = source()->depth();
  return depth ? depth - 1 : 0;
}

uint32_t SGraph::DeviceCount() const {
  std::unordered_set<DeviceNode*> devices;
  children_arc_mgr().Output(this, [&](Node* node) {
    DeviceNode* device = nullptr;
    device_arc_mgr().Output(node, &device);
    devices.insert(device);
  });
  return devices.size();
}

void SGraph::WalkArcReverse(const std::function<void(Arc<Node>*)>& cb) {
  WalkReverse([&](Node* node) {
    arc_mgr().OutputArc(node, [&](Arc<Node>* arc) { cb(arc); });
  });
}

void SGraph::WalkReverse(const std::function<void(Node*)>& cb) {
  auto next = std::unordered_set<Node*>{sink()};
  auto marked = std::unordered_set<Node*>{};
  while (next.size()) {
    auto queue = std::list<Node*>(next.begin(), next.end());
    for (const auto& node : queue) {
      cb(node);
      marked.insert(node);
      next.erase(node);
      arc_mgr().InputArc(node, [&](Arc<Node>* arc) {
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

void SGraph::WalkArc(const std::function<void(Arc<Node>*)>& cb) {
  Walk([&](Node* node) { arc_mgr().InputArc(node, cb); });
}

void SGraph::Walk(const std::function<void(Node*)>& cb) {
  auto next = std::unordered_set<Node*>{source()};
  auto marked = std::unordered_set<Node*>{};
  while (next.size()) {
    auto queue = std::list<Node*>(next.begin(), next.end());
    for (const auto& node : queue) {
      cb(node);
      marked.insert(node);
      next.erase(node);
      arc_mgr().OutputArc(node, [&](Arc<Node>* arc) {
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

void SGraph::InitAscendentArc() {
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

void SGraph::ForeachAscendent(Node* node,
                              const std::function<void(Node*)>& cb) const {
  ascendent_arc_mgr().Output(node, cb);
}

void SGraph::ForeachDescendent(Node* node,
                               const std::function<void(Node*)>& cb) const {
  ascendent_arc_mgr().Input(node, cb);
}

void SGraph::InitDepth() {
  WalkReverse([&](Node* node) {
    int depth = -1;
    arc_mgr().Output(node,
                     [&](Node* to) { depth = std::max(depth, to->depth()); });
    node->mut_depth() = depth + 1;
  });
}

}  // namespace schedule
}  // namespace oneflow
