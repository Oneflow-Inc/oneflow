/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/bfs_visitor.h"

namespace oneflow {
namespace schedule {

void SGraph::InitSourceAndSink() {
  mut_source() = mut_fake_node_mgr().Create("source");
  mut_sink() = mut_fake_node_mgr().Create("sink");
}

uint32_t SGraph::LossNodes(std::list<STask*>* l) const {
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
  auto foreach_next = std::bind(&SGraph::ForeachPrev, this,
                                std::placeholders::_1, std::placeholders::_2);
  auto foreach_prev = std::bind(&SGraph::ForeachNext, this,
                                std::placeholders::_1, std::placeholders::_2);
  BfsVisitor<STask*> bfs_foreach(foreach_next, foreach_prev);
  bfs_foreach(sink(), cb);
}

void SGraph::WalkArc(const std::function<void(Arc<STask>*)>& cb) {
  Walk([&](STask* node) { arc_mgr().InputArc(node, cb); });
}

void SGraph::Walk(const std::function<void(STask*)>& cb) {
  auto foreach_next = std::bind(&SGraph::ForeachNext, this,
                                std::placeholders::_1, std::placeholders::_2);
  auto foreach_prev = std::bind(&SGraph::ForeachPrev, this,
                                std::placeholders::_1, std::placeholders::_2);
  BfsVisitor<STask*> bfs_foreach(foreach_next, foreach_prev);
  bfs_foreach(source(), cb);
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
    uint32_t depth = 0;
    arc_mgr().Output(node, [&](STask* dst_node) {
      depth = std::max(depth, dst_node->depth());
    });
    node->mut_depth() = depth + 1;
  });
}

void SGraph::UpdateTask() {
  ForeachNode([&](STask* task) {
    SDevice* device = nullptr;
    device_arc_mgr().Output(task, &device);
    task->mut_device() = device;
  });
}

void SGraph::UpdateRegstDesc() {
  ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    STask* task = nullptr;
    produced_regst_desc_mgr().Input(regst_desc, &task);
    regst_desc->mut_owner_task() = task;
  });
}

}  // namespace schedule
}  // namespace oneflow
