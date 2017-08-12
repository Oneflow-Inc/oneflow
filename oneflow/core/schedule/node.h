/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_NODE_H_
#define ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_NODE_H_

#include <limits.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

// static schedule node
class SNode {
 public:
  explicit SNode(const std::string name) : name_(name) {}
  SNode() {}
  virtual ~SNode() {}

  virtual std::string name() const { return name_; }
  virtual std::string& mut_name() { return name_; }
  virtual uint64_t id() const { return id_; }
  virtual uint64_t& mut_id() { return id_; }

  DEFINE_METHOD_TYPE();

 protected:
  uint64_t id_;
  std::string name_;
};

template<typename NodeType = SNode>
class NodeMgr {
 public:
  NodeMgr() {}

  template<typename... Args>
  NodeType* Create(Args&&... args) {
    auto node =
        std::unique_ptr<NodeType>(new NodeType(std::forward<Args>(args)...));
    auto ret = node.get();
    do { node->mut_id() = GetAutoIncrementId(); } while (Find(node->id()));
    if (!Insert(std::move(node))) { ret = nullptr; }
    return ret;
  }

  template<typename... Args>
  NodeType* CreateWithId(uint64_t id, Args&&... args) {
    if (Find(id)) { return nullptr; }
    auto node =
        std::unique_ptr<NodeType>(new NodeType(std::forward<Args>(args)...));
    node->mut_id() = id;
    auto ret = node.get();
    if (!Insert(std::move(node))) { ret = nullptr; }
    return ret;
  }

  template<typename... Args>
  NodeType* CreateIfNotFound(const std::string& name, Args&&... args) {
    NodeType* node = Find(name);
    NodeType* ret = nullptr;
    if (node) {
      ret = dynamic_cast<NodeType*>(node);
    } else {
      ret = Create(name, std::forward<Args>(args)...);
    }
    return ret;
  }

  template<typename... Args>
  NodeType* CreateIfNotFound(uint64_t id, Args&&... args) {
    NodeType* node = Find(id);
    return node ? node : CreateWithId(id, std::forward<Args>(args)...);
  }

  int Insert(std::unique_ptr<NodeType>&& node) {
    if (Find(node->id())) { return 0; }
    name2id2node_[node->name()][node->id()] = node.get();
    id2node_[node->id()] = std::move(node);
    return 1;
  }

  NodeType* Find(const std::string& name) const {
    NodeType* node = nullptr;
    Find(name, [&](NodeType* ptr) { node = ptr; });
    return node;
  }

  int Find(const std::string& name,
           const std::function<void(NodeType*)>& cb) const {
    auto itt = name2id2node_.find(name);
    int count = 0;
    if (itt == name2id2node_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  NodeType* Find(uint64_t id) const {
    auto ret = id2node_.find(id);
    if (id2node_.end() == ret) { return nullptr; }
    return ret->second.get();
  }

  void Delete(uint64_t id) {
    auto node = Find(id);
    if (node) {
      name2id2node_[node->name()].erase(id);
      id2node_.erase(id);
    }
  }

  void Dump() const {
    std::cout << "id2node_.size(): " << id2node_.size() << std::endl;
    std::cout << "name2id2node_.size(): " << name2id2node_.size() << std::endl;
  }

 private:
  std::unordered_map<uint64_t, std::unique_ptr<NodeType>> id2node_;
  std::unordered_map<std::string, std::unordered_map<uint64_t, NodeType*>>
      name2id2node_;
};

template<typename SrcNodeType, typename DstNodeType = SrcNodeType>
class Arc : public SNode {
 public:
  Arc(SrcNodeType* from, DstNodeType* to) {
    mut_from() = from;
    mut_to() = to;
    mut_name() = "";
  }

  Arc(SrcNodeType* from, DstNodeType* to, const std::string& name) {
    mut_from() = from;
    mut_to() = to;
    mut_name() = name;
  }

  typedef SrcNodeType src_node_type;
  typedef DstNodeType dst_node_type;

  SrcNodeType* from() const { return from_; }
  SrcNodeType*& mut_from() { return from_; }

  DstNodeType* to() const { return to_; }
  DstNodeType*& mut_to() { return to_; }

  std::string name() const { return from()->name() + "->" + to()->name(); }

 private:
  SrcNodeType* from_;
  DstNodeType* to_;
};

template<typename ArcType,
         typename SrcNodeType = typename ArcType::src_node_type,
         typename DstNodeType = typename ArcType::dst_node_type>
class ArcMgr {
 public:
  ArcMgr() = default;
  virtual ~ArcMgr() = default;

  template<typename... Args>
  ArcType* CreateIfNotFound(SrcNodeType* from, DstNodeType* to,
                            Args&&... args) {
    auto node = Find(from, to);
    return node ? node : Create(from, to, std::forward<Args>(args)...);
  }

  int Insert(std::unique_ptr<ArcType>&& arcp) {
    if (Find(arcp->from(), arcp->to())) { return 0; }
    from2to2arc_[arcp->from()][arcp->to()] = arcp.get();
    to2from2arc_[arcp->to()][arcp->from()] = arcp.get();
    id2arc_[arcp->id()] = std::move(arcp);
    return 1;
  }

  unsigned int Input(const DstNodeType* to) const {
    return Input(to, [](SrcNodeType* from) {});
  }

  unsigned int Input(const DstNodeType* to, std::list<SrcNodeType*>* l) const {
    return Input(to, [l](SrcNodeType* from) { l->push_back(from); });
  }

  unsigned int Input(const DstNodeType* to,
                     const std::function<void(SrcNodeType*)>& cb) const {
    return InputArc(to, [&cb](ArcType* arcp) { cb(arcp->from()); });
  }

  unsigned int Input(const DstNodeType* to, SrcNodeType** from) const {
    return InputArc(to, [from](ArcType* p) { *from = p->from(); });
  }

  unsigned int InputArc(const DstNodeType* to, std::list<ArcType*>* l) const {
    return InputArc(to, [l](ArcType* arc) { l->push_back(arc); });
  }

  unsigned int InputArc(std::list<DstNodeType*> to_nodes,
                        const std::function<void(ArcType*)>& cb) const {
    unsigned int count = 0;
    for (auto node_itt = to_nodes.begin(); node_itt != to_nodes.end();
         node_itt++) {
      auto itt = to2from2arc_.find(*node_itt);
      if (itt == to2from2arc_.end()) { continue; }
      for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
        cb(jtt->second);
        count++;
      }
    }
    return count;
  }

  unsigned int InputArc(const DstNodeType* to, ArcType* ptr) const {
    return InputArc(to, [&ptr](ArcType* p) { ptr = p; });
  }

  unsigned int InputArc(const DstNodeType* to,
                        const std::function<void(ArcType*)>& cb) const {
    unsigned int count = 0;
    auto itt = to2from2arc_.find(const_cast<DstNodeType*>(to));
    if (itt == to2from2arc_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  unsigned int Output(const SrcNodeType* from, DstNodeType** to) const {
    return OutputArc(from, [&to](ArcType* p) { *to = p->to(); });
  }

  unsigned int Output(const SrcNodeType* from) const {
    return Output(from, [](DstNodeType* to) {});
  }

  unsigned int Output(const SrcNodeType* from,
                      std::list<DstNodeType*>* l) const {
    return Output(from, [&l](DstNodeType* to) { l->push_back(to); });
  }

  unsigned int Output(const SrcNodeType* from,
                      const std::function<void(DstNodeType*)>& cb) const {
    return OutputArc(from, [&cb](ArcType* arcp) { cb(arcp->to()); });
  }

  unsigned int OutputArc(const SrcNodeType* from, ArcType* ptr) const {
    return OutputArc(from, [&ptr](ArcType* p) { ptr = p; });
  }

  unsigned int OutputArc(const SrcNodeType* from,
                         std::list<ArcType*>* l) const {
    return OutputArc(from, [&l](ArcType* arc) { l->push_back(arc); });
  }

  unsigned int OutputArc(const SrcNodeType* from,
                         std::function<void(ArcType*)> cb) const {
    unsigned int count = 0;
    auto itt = from2to2arc_.find(const_cast<SrcNodeType*>(from));
    if (itt == from2to2arc_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  unsigned int OutputArc(const std::list<SrcNodeType*>& from_nodes,
                         const std::function<void(ArcType*)>& cb) const {
    unsigned int count = 0;
    for (auto node_itt = from_nodes.begin(); node_itt != from_nodes.end();
         node_itt++) {
      auto itt = from2to2arc_.find(*node_itt);
      if (itt == from2to2arc_.end()) { continue; }
      for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
        cb(jtt->second);
        count++;
      }
    }
    return count;
  }

  void Find(const std::list<SrcNodeType*>& from_nodes,
            const std::list<DstNodeType*>& to_nodes,
            const std::function<void(ArcType*)>& cb) const {
    for (auto from : from_nodes) {
      auto itt = from2to2arc_.find(from);
      if (itt == from2to2arc_.end()) { continue; }
      for (auto to : to_nodes) {
        auto jtt = itt->second.find(to);
        if (jtt == itt->second.end()) { continue; }
        cb(jtt->second);
      }
    }
  }

  ArcType* Find(SrcNodeType* from, DstNodeType* to) const {
    auto itt = from2to2arc_.find(from);
    if (itt == from2to2arc_.end()) { return nullptr; }
    auto jtt = itt->second.find(to);
    if (jtt == itt->second.end()) { return nullptr; }
    return jtt->second;
  }

  ArcType* Find(uint64_t id) const {
    auto itt = id2arc_.find(id);
    if (itt == id2arc_.end()) { return nullptr; }
    return itt->second.get();
  }

  void Delete(uint64_t id) {
    auto arcp = Find(id);
    if (!arcp) { return; }
    from2to2arc_[arcp->from()].erase(arcp->to());
    if (!from2to2arc_[arcp->from()].size()) {
      from2to2arc_.erase(arcp->from());
    }
    to2from2arc_[arcp->to()].erase(arcp->from());
    if (!to2from2arc_[arcp->to()].size()) { to2from2arc_.erase(arcp->to()); }
    id2arc_.erase(id);
  }

  void Delete(SrcNodeType* from, DstNodeType* to) {
    auto arcp = Find(from, to);
    if (!arcp) { return; }
    Delete(arcp->id());
  }

  void Dump() const {
    std::cout << "from2to2arc_.size() = " << from2to2arc_.size() << std::endl;
    std::cout << "to2from2arc_.size() = " << to2from2arc_.size() << std::endl;
    std::cout << "id2arc_.size() = " << id2arc_.size() << std::endl;
  }

  template<typename... Args>
  ArcType* Create(Args&&... args) {
    auto p = std::unique_ptr<ArcType>(new ArcType(std::forward<Args>(args)...));
    auto ret = p.get();
    p->mut_id() = GetAutoIncrementId();
    if (!Insert(std::move(p))) { ret = nullptr; }
    return ret;
  }

 protected:
  std::unordered_map<SrcNodeType*, std::unordered_map<DstNodeType*, ArcType*>>
      from2to2arc_;
  std::unordered_map<DstNodeType*, std::unordered_map<SrcNodeType*, ArcType*>>
      to2from2arc_;
  std::unordered_map<uint64_t, std::unique_ptr<ArcType>> id2arc_;
};

template<typename ArcType,
         typename SrcNodeType = typename ArcType::src_node_type,
         typename DstNodeType = typename ArcType::dst_node_type>
class HasOneArcMgr : public ArcMgr<ArcType> {
 public:
  HasOneArcMgr() {}

  template<typename... Args>
  ArcType* CreateIfNotFound(SrcNodeType* from, DstNodeType* to,
                            Args&&... args) {
    std::list<DstNodeType*> to_nodes;
    this->Output(from, &to_nodes);

    for (auto node : to_nodes) { this->Delete(from, node); }
    return ArcMgr<ArcType>::CreateIfNotFound(from, to,
                                             std::forward<Args>(args)...);
  }
};

//	static schedule device
class SDevice : public SNode {
 public:
  SDevice(std::string name, unsigned int time) : SNode(name), time_(time) {}
  unsigned int time() const { return time_; }
  unsigned int& mut_time() { return time_; }

  uint64_t memory_limit() const { return memory_limit_; }
  uint64_t& mut_memory_limit() { return memory_limit_; }

  DEFINE_METHOD_TYPE();

 private:
  unsigned int time_;
  uint64_t memory_limit_ = ULLONG_MAX;
};

class SRegstDesc : public SNode {
 public:
  SRegstDesc(const std::string name) : SNode(name) {}
  SRegstDesc() : SNode() {}
  virtual ~SRegstDesc() {}

  uint64_t regst_size() const { return regst_size_; }
  uint64_t& mut_regst_size() { return regst_size_; }

 private:
  uint64_t regst_size_ = 1u;
};

//	static schedule task

class STask : public SNode {
 public:
  explicit STask(const std::string name) : SNode(name) {}
  STask() {}
  virtual ~STask() {}
  DEFINE_METHOD_TYPE();
  inline int depth() const { return depth_; }
  inline int& mut_depth() { return depth_; }

 protected:
  int depth_;
  SDevice* device_;
};

//	static schedule graph

class SGraph : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SGraph);
  SGraph() = default;
  ~SGraph() = default;
  DEFINE_METHOD_TYPE();

  explicit SGraph(std::string name) : SNode(name) { InitSourceAndSink(); }

  static std::unique_ptr<SGraph> CreateFromPlan(const Plan& plan) {
    return unique_ptr_new<SGraph>("plan");
  }

  void Update() {
    UpdateSourceAndSink();
    InitDepth();
    InitAscendentArc();
  }

  void InitSourceAndSink();
  void InitDepth();

  static bool DescNodeOrder(STask* a, STask* b) {
    return a->depth() < b->depth();
  }

  static bool AscNodeOrder(STask* a, STask* b) {
    return a->depth() > b->depth();
  }

  void InitAscendentArc();

  void ForeachNode(const std::function<void(STask*)>& cb) const;
  void ForeachAscendent(STask*, const std::function<void(STask*)>& cb) const;
  void ForeachDescendent(STask*, const std::function<void(STask*)>& cb) const;
  void ForeachNodeWithSourceAndSink(
      const std::function<void(STask*)>& cb) const;
  void ForeachRegstDesc(const std::function<void(SRegstDesc*)>& cb) const;

  void Walk(const std::function<void(STask*)>& cb);
  void WalkArc(const std::function<void(Arc<STask>*)>& cb);
  uint32_t DeviceCount() const;
  uint32_t Depth() const;
  void WalkReverse(const std::function<void(STask*)>& cb);
  void WalkArcReverse(const std::function<void(Arc<STask>*)>& cb);
  void ForeachArc(const std::function<void(Arc<STask>*)>& cb) const;
  void UpdateSourceAndSink();
  int LossNodes(std::list<STask*>* l) const;
  STask* source() const { return source_; }
  STask*& mut_source() { return source_; }

  STask* sink() const { return sink_; }
  STask*& mut_sink() { return sink_; }

  inline const NodeMgr<STask>& node_mgr() const { return node_mgr_; }
  inline NodeMgr<STask>& mut_node_mgr() { return node_mgr_; }

  inline NodeMgr<STask>& mut_fake_node_mgr() { return fake_node_mgr_; }

  inline const ArcMgr<Arc<STask>>& arc_mgr() const { return arc_mgr_; }
  inline ArcMgr<Arc<STask>>& mut_arc_mgr() { return arc_mgr_; }

  inline const HasOneArcMgr<Arc<STask, SDevice>>& device_arc_mgr() const {
    return device_arc_mgr_;
  }
  inline HasOneArcMgr<Arc<STask, SDevice>>& mut_device_arc_mgr() {
    return device_arc_mgr_;
  }

  inline NodeMgr<SDevice>& mut_device_mgr() { return device_mgr_; }

  inline const ArcMgr<Arc<SGraph, STask>>& loss_arc_mgr() const {
    return loss_arc_mgr_;
  }
  inline ArcMgr<Arc<SGraph, STask>>& mut_loss_arc_mgr() {
    return loss_arc_mgr_;
  }

  inline const ArcMgr<Arc<STask>>& ascendent_arc_mgr() const {
    return ascendent_arc_mgr_;
  }
  inline ArcMgr<Arc<STask>>& mut_ascendent_arc_mgr() {
    return ascendent_arc_mgr_;
  }

  inline const ArcMgr<Arc<SGraph, STask>>& children_arc_mgr() const {
    return children_arc_mgr_;
  }
  inline ArcMgr<Arc<SGraph, STask>>& mut_children_arc_mgr() {
    return children_arc_mgr_;
  }

  inline NodeMgr<SRegstDesc>& mut_regst_desc_mgr() { return regst_desc_mgr_; }

  inline const ArcMgr<Arc<STask, SRegstDesc>>& produced_regst_desc_mgr() const {
    return produced_regst_desc_mgr_;
  }
  inline ArcMgr<Arc<STask, SRegstDesc>>& mut_produced_regst_desc_mgr() {
    return produced_regst_desc_mgr_;
  }

  inline const ArcMgr<Arc<STask, SRegstDesc>>& subscribed_regst_desc_mgr()
      const {
    return subscribed_regst_desc_mgr_;
  }
  inline ArcMgr<Arc<STask, SRegstDesc>>& mut_subscribed_regst_desc_mgr() {
    return subscribed_regst_desc_mgr_;
  }

 private:
  STask* source_;
  STask* sink_;
  NodeMgr<STask> node_mgr_;
  NodeMgr<STask> fake_node_mgr_;
  NodeMgr<SRegstDesc> regst_desc_mgr_;
  NodeMgr<SDevice> device_mgr_;
  ArcMgr<Arc<STask>> arc_mgr_;
  ArcMgr<Arc<SGraph, STask>> loss_arc_mgr_;
  ArcMgr<Arc<SGraph, STask>> children_arc_mgr_;
  ArcMgr<Arc<STask>> ascendent_arc_mgr_;
  ArcMgr<Arc<STask, SRegstDesc>> produced_regst_desc_mgr_;
  ArcMgr<Arc<STask, SRegstDesc>> subscribed_regst_desc_mgr_;
  HasOneArcMgr<Arc<STask, SDevice>> device_arc_mgr_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_NODE_H_
