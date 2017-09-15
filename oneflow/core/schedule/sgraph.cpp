#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/bfs_visitor.h"

namespace oneflow {
namespace schedule {

std::string SGraph::ToDotString() {
  std::stringstream ss;
  ss << "digraph {" << std::endl;
  ForEachArc([&](TaskArc* arc) {
    ss << "\t\"" << arc->src_node()->name() << "\" -> \""
       << arc->dst_node()->name() << "\";" << std::endl;
  });
  ss << "}" << std::endl;
  return ss.str();
}

void SGraph::InitSourceAndSink() {
  mut_source() = mut_fake_node_mgr()->Create("source");
  mut_sink() = mut_fake_node_mgr()->Create("sink");
  SDevice* device = mut_device_mgr()->Create("fake_device");
  mut_device_arc_mgr()->CreateIfNotFound(source(), device);
  mut_device_arc_mgr()->CreateIfNotFound(sink(), device);
}

uint32_t SGraph::LossNodes(std::list<STask*>* l) const {
  return loss_arc_mgr().Output(this, l);
}

void SGraph::UpdateSourceAndSink() {
  std::list<Arc<STask>*> arcs;
  arc_mgr().OutputArc(source(), &arcs);
  arc_mgr().InputArc(sink(), &arcs);
  for (TaskArc* arc : arcs) { mut_arc_mgr()->Delete(arc->id()); }
  children_arc_mgr().Output(this, [&](STask* leaf) {
    if (!arc_mgr().Input(leaf)) {
      mut_arc_mgr()->CreateIfNotFound(source(), leaf);
    }
    if (!arc_mgr().Output(leaf)) {
      mut_arc_mgr()->CreateIfNotFound(leaf, sink());
    }
  });
}

bool SGraph::ReachableWithoutArc(const TaskArc* arc) const {
  bool reachable = false;
  arc_mgr().Input(arc->dst_node(), [&](STask* prev) {
    TaskArc* asc = ascendant_arc_mgr().Find(prev, arc->src_node());
    reachable = reachable || asc != nullptr;
  });
  return reachable;
}

void SGraph::RemoveUselessArc() {
  std::unordered_set<uint64_t> useless_arc_ids;
  ForEachArc([&](TaskArc* arc) {
    if (ReachableWithoutArc(arc)) useless_arc_ids.insert(arc->id());
  });
  for (uint64_t id : useless_arc_ids) { mut_arc_mgr()->Delete(id); }
}

void SGraph::ForEachArc(const std::function<void(Arc<STask>*)>& cb) const {
  ForEachNode([&](STask* child) { arc_mgr().OutputArc(child, cb); });
}

void SGraph::ForEachChild(const std::function<void(const STask&)>& cb) const {
  children_arc_mgr().Output(this, cb);
}

void SGraph::MutForEachChild(const std::function<void(STask*)>& cb) const {
  children_arc_mgr().Output(this, cb);
}

void SGraph::ForEachNode(const std::function<void(STask*)>& cb) const {
  cb(source());
  children_arc_mgr().Output(this, cb);
  cb(sink());
}

void SGraph::ForEachNode(const std::function<void(const STask&)>& cb) const {
  ForEachNode([&](STask* task) { cb(*task); });
}

void SGraph::ForEachRegstDesc(
    const std::function<void(SRegstDesc*)>& cb) const {
  children_arc_mgr().Output(
      this, [&](STask* node) { produced_regst_desc_mgr().Output(node, cb); });
}

uint32_t SGraph::Depth() const {
  uint32_t depth = source()->depth();
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

void SGraph::WalkArcReverse(const std::function<void(Arc<STask>*)>& cb) const {
  WalkReverse([&](STask* node) {
    arc_mgr().OutputArc(node, [&](Arc<STask>* arc) { cb(arc); });
  });
}

void SGraph::WalkReverse(const std::function<void(STask*)>& cb) const {
  auto foreach_next = std::bind(&SGraph::ForEachPrev, this,
                                std::placeholders::_1, std::placeholders::_2);
  auto foreach_prev = std::bind(&SGraph::ForEachNext, this,
                                std::placeholders::_1, std::placeholders::_2);
  BfsVisitor<STask*> bfs_foreach(foreach_next, foreach_prev);
  bfs_foreach(sink(), cb);
}

void SGraph::WalkArc(const std::function<void(Arc<STask>*)>& cb) const {
  Walk([&](STask* node) { arc_mgr().InputArc(node, cb); });
}

void SGraph::Walk(const std::function<void(STask*)>& cb) const {
  auto foreach_next = std::bind(&SGraph::ForEachNext, this,
                                std::placeholders::_1, std::placeholders::_2);
  auto foreach_prev = std::bind(&SGraph::ForEachPrev, this,
                                std::placeholders::_1, std::placeholders::_2);
  BfsVisitor<STask*> bfs_foreach(foreach_next, foreach_prev);
  bfs_foreach(source(), cb);
}

void SGraph::InitAscendantArc() {
  Walk([&](STask* node) {
    arc_mgr().Input(node, [&](STask* prev) {
      std::list<STask*> l;
      ascendant_arc_mgr().Output(prev, &l);
      for (STask* asc : l) {
        mut_ascendant_arc_mgr()->CreateIfNotFound(node, asc);
      }
      mut_ascendant_arc_mgr()->CreateIfNotFound(node, prev);
    });
  });
}

void SGraph::ForEachAscendant(STask* node,
                              const std::function<void(STask*)>& cb) const {
  ascendant_arc_mgr().Output(node, cb);
}

void SGraph::ForEachDescendant(STask* node,
                               const std::function<void(STask*)>& cb) const {
  ascendant_arc_mgr().Input(node, cb);
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
  ForEachNode([&](STask* task) {
    SDevice* device = nullptr;
    device_arc_mgr().Output(task, &device);
    task->mut_device() = device;
  });
}

void SGraph::UpdateRegstDesc() {
  ForEachRegstDesc([&](SRegstDesc* regst_desc) {
    STask* task = nullptr;
    produced_regst_desc_mgr().Input(regst_desc, &task);
    regst_desc->mut_owner_task() = task;
  });
}

}  // namespace schedule
}  // namespace oneflow
