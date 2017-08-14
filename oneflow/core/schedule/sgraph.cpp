/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/sgraph.h"

namespace oneflow {
namespace schedule {

void SGraph::InitSourceAndSink() {
  mut_source() = mut_fake_node_mgr().Create("source");
  mut_sink() = mut_fake_node_mgr().Create("sink");
}

int SGraph::LossNodes(std::list<STask*>* l) const {
  return loss_arc_mgr().Output(this, l);
}

void SGraph::UpdateSourceAndSink() {
  std::list<Arc<STask>*> arcs;
  arc_mgr().OutputArc(source(), &arcs);
  arc_mgr().InputArc(sink(), &arcs);
  for (auto arc : arcs) { mut_arc_mgr().Delete(arc->id()); }
  children_arc_mgr().Output(this, [&](STask* leaf) {
    if (!arc_mgr().Input(leaf)) {
      mut_arc_mgr().CreateIfNotFound(source(), leaf);
    }
    if (!arc_mgr().Output(leaf)) {
      mut_arc_mgr().CreateIfNotFound(leaf, sink());
    }
  });
}

void SGraph::ForeachArc(const std::function<void(Arc<STask>*)>& cb) const {
  arc_mgr().OutputArc(source(), cb);
  children_arc_mgr().Output(
      this, [&](STask* child) { arc_mgr().OutputArc(child, cb); });
}

void SGraph::ForeachNodeWithSourceAndSink(
    const std::function<void(STask*)>& cb) const {
  cb(source());
  ForeachNode(cb);
  cb(sink());
}
void SGraph::ForeachNode(const std::function<void(STask*)>& cb) const {
  cb(source());
  children_arc_mgr().Output(this, cb);
  cb(sink());
}

void SGraph::ForeachRegstDesc(
    const std::function<void(SRegstDesc*)>& cb) const {
  children_arc_mgr().Output(
      this, [&](STask* node) { produced_regst_desc_mgr().Output(node, cb); });
}

uint32_t SGraph::Depth() const {
  auto depth = source()->depth();
  return depth ? depth - 1 : 0;
}

uint32_t SGraph::DeviceCount() const {
  std::unordered_set<SDevice*> devices;
  children_arc_mgr().Output(this, [&](STask* node) {
    SDevice* device = nullptr;
    device_arc_mgr().Output(node, &device);
    devices.insert(device);
  });
  return devices.size();
}

void SGraph::WalkArcReverse(const std::function<void(Arc<STask>*)>& cb) {
  WalkReverse([&](STask* node) {
    arc_mgr().OutputArc(node, [&](Arc<STask>* arc) { cb(arc); });
  });
}

void SGraph::WalkReverse(const std::function<void(STask*)>& cb) {
  auto next = std::unordered_set<STask*>{sink()};
  auto marked = std::unordered_set<STask*>{};
  while (next.size()) {
    auto queue = std::list<STask*>(next.begin(), next.end());
    for (const auto& node : queue) {
      cb(node);
      marked.insert(node);
      next.erase(node);
      arc_mgr().InputArc(node, [&](Arc<STask>* arc) {
        bool all_marked = true;
        arc_mgr().Output(arc->from(), [&](STask* from) {
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

void SGraph::WalkArc(const std::function<void(Arc<STask>*)>& cb) {
  Walk([&](STask* node) { arc_mgr().InputArc(node, cb); });
}

void SGraph::Walk(const std::function<void(STask*)>& cb) {
  auto next = std::unordered_set<STask*>{source()};
  auto marked = std::unordered_set<STask*>{};
  while (next.size()) {
    auto queue = std::list<STask*>(next.begin(), next.end());
    for (const auto& node : queue) {
      cb(node);
      marked.insert(node);
      next.erase(node);
      arc_mgr().OutputArc(node, [&](Arc<STask>* arc) {
        bool all_marked = true;
        arc_mgr().Input(arc->to(), [&](STask* from) {
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
  Walk([&](STask* node) {
    arc_mgr().Input(node, [&](STask* prev) {
      std::list<STask*> l;
      ascendent_arc_mgr().Output(prev, &l);
      for (STask* asc : l) {
        mut_ascendent_arc_mgr().CreateIfNotFound(node, asc);
      }
      mut_ascendent_arc_mgr().CreateIfNotFound(node, prev);
    });
  });
}

void SGraph::ForeachAscendent(STask* node,
                              const std::function<void(STask*)>& cb) const {
  ascendent_arc_mgr().Output(node, cb);
}

void SGraph::ForeachDescendent(STask* node,
                               const std::function<void(STask*)>& cb) const {
  ascendent_arc_mgr().Input(node, cb);
}

void SGraph::InitDepth() {
  WalkReverse([&](STask* node) {
    int depth = -1;
    arc_mgr().Output(node,
                     [&](STask* to) { depth = std::max(depth, to->depth()); });
    node->mut_depth() = depth + 1;
  });
}

}  // namespace schedule
}  // namespace oneflow
