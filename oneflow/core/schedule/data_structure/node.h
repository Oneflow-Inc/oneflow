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

#include "oneflow/core/schedule/util/util.h"

namespace oneflow {
namespace schedule {

class Node {
 public:
  explicit Node(const std::string name) : name_(name) {}
  Node() {}
  virtual ~Node() {}

  virtual std::string name() const { return name_; }
  virtual std::string& mut_name() { return name_; }
  virtual uint64_t id() const { return id_; }
  virtual uint64_t& mut_id() { return id_; }
  int depth() const { return depth_; }
  int& mut_depth() { return depth_; }

  DEFINE_METHOD_TYPE();

 protected:
  uint64_t id_;
  std::string name_;
  int depth_;
  bool is_source_;
  bool is_sink_;
};

template<typename T = Node>
class NodeMgr {
 public:
  NodeMgr() {}

  template<typename... Args>
  T* Create(Args&&... args) {
    auto node = std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    auto ret = node.get();
    do { node->mut_id() = GetAutoIncrementId(); } while (Find(node->id()));
    if (!Insert(std::move(node))) { ret = nullptr; }
    return ret;
  }

  template<typename... Args>
  T* CreateWithId(uint64_t id, Args&&... args) {
    if (Find(id)) { return nullptr; }
    auto node = std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    node->mut_id() = id;
    auto ret = node.get();
    if (!Insert(std::move(node))) { ret = nullptr; }
    return ret;
  }

  template<typename... Args>
  T* CreateIfNotFound(const std::string& name, Args&&... args) {
    Node* node = Find(name);
    T* ret = nullptr;
    if (node) {
      ret = dynamic_cast<T*>(node);
    } else {
      ret = Create(name, std::forward<Args>(args)...);
    }
    return ret;
  }

  template<typename... Args>
  T* CreateIfNotFound(uint64_t id, Args&&... args) {
    Node* node = Find(id);
    T* ret = nullptr;
    if (node) {
      ret = dynamic_cast<T*>(node);
    } else {
      ret = CreateWithId(id, std::forward<Args>(args)...);
    }
    return ret;
  }

  int Insert(std::unique_ptr<Node>&& node) {
    if (Find(node->id())) { return 0; }
    name2id2node_[node->name()][node->id()] = node.get();
    id2node_[node->id()] = std::move(node);
    return 1;
  }

  Node* Find(const std::string& name) const {
    Node* node = nullptr;
    Find(name, [&](Node* ptr) { node = ptr; });
    return node;
  }

  int Find(const std::string& name,
           const std::function<void(Node*)>& cb) const {
    auto itt = name2id2node_.find(name);
    int count = 0;
    if (itt == name2id2node_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  Node* Find(uint64_t id) const {
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
  std::unordered_map<uint64_t, std::unique_ptr<Node>> id2node_;
  std::unordered_map<std::string, std::unordered_map<uint64_t, Node*>>
      name2id2node_;
};

class Arc : public Node {
 public:
  Arc(Node* from, Node* to) {
    mut_from() = from;
    mut_to() = to;
    mut_name() = "";
  }

  Arc(Node* from, Node* to, const std::string& name) {
    mut_from() = from;
    mut_to() = to;
    mut_name() = name;
  }

  Node* from() const { return from_; }
  Node*& mut_from() { return from_; }

  Node* to() const { return to_; }
  Node*& mut_to() { return to_; }

  std::string name() const { return from()->name() + "->" + to()->name(); }

 private:
  Node* from_;
  Node* to_;
};

class ArcMgr {
 public:
  ArcMgr() {}

  template<typename... Args>
  Arc* CreateIfNotFound(Node* from, Node* to, Args&&... args) {
    auto node = Find(from, to);
    return node ? node : Create(from, to, std::forward<Args>(args)...);
  }

  int Insert(std::unique_ptr<Arc>&& arcp) {
    if (Find(arcp->from(), arcp->to())) { return 0; }
    from2to2arc_[arcp->from()][arcp->to()] = arcp.get();
    to2from2arc_[arcp->to()][arcp->from()] = arcp.get();
    id2arc_[arcp->id()] = std::move(arcp);
    return 1;
  }

  unsigned int Input(const Node* to) const {
    return Input(to, [](Node* from) {});
  }

  unsigned int Input(const Node* to, std::list<Node*>* l) const {
    return Input(to, [l](Node* from) { l->push_back(from); });
  }

  unsigned int Input(const Node* to,
                     const std::function<void(Node*)>& cb) const {
    return InputArc(to, [&cb](Arc* arcp) { cb(arcp->from()); });
  }

  unsigned int Input(const Node* to, Node** from) const {
    return InputArc(to, [from](Arc* p) { *from = p->from(); });
  }

  unsigned int InputArc(const Node* to, std::list<Arc*>* l) const {
    return InputArc(to, [l](Arc* arc) { l->push_back(arc); });
  }

  unsigned int InputArc(std::list<Node*> to_nodes,
                        const std::function<void(Arc*)>& cb) const {
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

  unsigned int InputArc(const Node* to, Arc* ptr) const {
    return InputArc(to, [&ptr](Arc* p) { ptr = p; });
  }

  unsigned int InputArc(const Node* to,
                        const std::function<void(Arc*)>& cb) const {
    unsigned int count = 0;
    auto itt = to2from2arc_.find(const_cast<Node*>(to));
    if (itt == to2from2arc_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  unsigned int Output(const Node* from, Node** to) const {
    return OutputArc(from, [&to](Arc* p) { *to = p->to(); });
  }

  unsigned int Output(const Node* from) const {
    return Output(from, [](Node* to) {});
  }

  unsigned int Output(const Node* from, std::list<Node*>* l) const {
    return Output(from, [&l](Node* to) { l->push_back(to); });
  }

  unsigned int Output(const Node* from,
                      const std::function<void(Node*)>& cb) const {
    return OutputArc(from, [&cb](Arc* arcp) { cb(arcp->to()); });
  }

  unsigned int OutputArc(const Node* from, Arc* ptr) const {
    return OutputArc(from, [&ptr](Arc* p) { ptr = p; });
  }

  unsigned int OutputArc(const Node* from, std::list<Arc*>* l) const {
    return OutputArc(from, [&l](Arc* arc) { l->push_back(arc); });
  }

  unsigned int OutputArc(const Node* from, std::function<void(Arc*)> cb) const {
    unsigned int count = 0;
    auto itt = from2to2arc_.find(const_cast<Node*>(from));
    if (itt == from2to2arc_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  unsigned int OutputArc(const std::list<Node*>& from_nodes,
                         const std::function<void(Arc*)>& cb) const {
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

  void Find(const std::list<Node*>& from_nodes,
            const std::list<Node*>& to_nodes,
            const std::function<void(Arc*)>& cb) const {
    for (Node* from : from_nodes) {
      auto itt = from2to2arc_.find(from);
      if (itt == from2to2arc_.end()) { continue; }
      for (Node* to : to_nodes) {
        auto jtt = itt->second.find(to);
        if (jtt == itt->second.end()) { continue; }
        cb(jtt->second);
      }
    }
  }

  Arc* Find(Node* from, Node* to) const {
    auto itt = from2to2arc_.find(from);
    if (itt == from2to2arc_.end()) { return nullptr; }
    auto jtt = itt->second.find(to);
    if (jtt == itt->second.end()) { return nullptr; }
    return jtt->second;
  }

  Arc* Find(uint64_t id) const {
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

  void Delete(Node* from, Node* to) {
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
  Arc* Create(Args&&... args) {
    auto p = std::unique_ptr<Arc>(new Arc(std::forward<Args>(args)...));
    auto ret = p.get();
    p->mut_id() = GetAutoIncrementId();
    if (!Insert(std::move(p))) { ret = nullptr; }
    return ret;
  }

 protected:
  std::unordered_map<Node*, std::unordered_map<Node*, Arc*>> from2to2arc_;
  std::unordered_map<Node*, std::unordered_map<Node*, Arc*>> to2from2arc_;
  std::unordered_map<uint64_t, std::unique_ptr<Arc>> id2arc_;
};

class HasOneArcMgr : public ArcMgr {
 public:
  HasOneArcMgr() {}

  template<typename... Args>
  Arc* CreateIfNotFound(Node* from, Node* to, Args&&... args) {
    std::list<Node*> to_nodes;
    Output(from, &to_nodes);

    for (auto node : to_nodes) { Delete(from, node); }
    return ArcMgr::CreateIfNotFound(from, to, std::forward<Args>(args)...);
  }
};

class DeviceNode : public Node {
 public:
  DeviceNode(std::string name, unsigned int time) : Node(name), time_(time) {}
  unsigned int time() const { return time_; }
  unsigned int& mut_time() { return time_; }

  uint64_t memory_limit() const { return memory_limit_; }
  uint64_t& mut_memory_limit() { return memory_limit_; }

  DEFINE_METHOD_TYPE();

 private:
  unsigned int time_;
  uint64_t memory_limit_ = ULLONG_MAX;
};

class GraphNode : public Node {
 public:
  DEFINE_METHOD_TYPE();

  GraphNode(std::string name) : Node(name) { InitSourceAndSink(); }

  void Update() {
    UpdateSourceAndSink();
    InitDepth();
    InitAscendentArc();
  }

  void InitSourceAndSink();
  void InitDepth();

  static bool DescNodeOrder(Node* a, Node* b) {
    return a->depth() < b->depth();
  }

  static bool AscNodeOrder(Node* a, Node* b) { return a->depth() > b->depth(); }

  void InitAscendentArc();

  void ForeachNode(const std::function<void(Node*)>& cb) const;
  void ForeachAscendent(Node*, const std::function<void(Node*)>& cb) const;
  void ForeachDescendent(Node*, const std::function<void(Node*)>& cb) const;
  void ForeachNodeWithSourceAndSink(const std::function<void(Node*)>& cb) const;
  void ForeachRegstDesc(const std::function<void(Node*)>& cb) const;

  void Walk(const std::function<void(Node*)>& cb);
  void WalkArc(const std::function<void(Arc*)>& cb);
  uint32_t DeviceCount() const;
  uint32_t Depth() const;
  void WalkReverse(const std::function<void(Node*)>& cb);
  void WalkArcReverse(const std::function<void(Arc*)>& cb);
  void ForeachArc(const std::function<void(Arc*)>& cb) const;
  void UpdateSourceAndSink();
  int LossNodes(std::list<Node*>* l) const;
  Node* source() const { return source_; }
  Node*& mut_source() { return source_; }

  Node* sink() const { return sink_; }
  Node*& mut_sink() { return sink_; }

  inline const NodeMgr<Node>& node_mgr() const { return node_mgr_; }
  inline NodeMgr<Node>& mut_node_mgr() { return node_mgr_; }

  inline NodeMgr<Node>& mut_fake_node_mgr() { return fake_node_mgr_; }

  inline const ArcMgr& arc_mgr() const { return arc_mgr_; }
  inline ArcMgr& mut_arc_mgr() { return arc_mgr_; }

  inline const HasOneArcMgr& device_arc_mgr() const { return device_arc_mgr_; }
  inline HasOneArcMgr& mut_device_arc_mgr() { return device_arc_mgr_; }

  inline NodeMgr<DeviceNode>& mut_device_mgr() { return device_mgr_; }

  inline const ArcMgr& loss_arc_mgr() const { return loss_arc_mgr_; }
  inline ArcMgr& mut_loss_arc_mgr() { return loss_arc_mgr_; }

  inline const ArcMgr& ascendent_arc_mgr() const { return ascendent_arc_mgr_; }
  inline ArcMgr& mut_ascendent_arc_mgr() { return ascendent_arc_mgr_; }

  inline const ArcMgr& children_arc_mgr() const { return children_arc_mgr_; }
  inline ArcMgr& mut_children_arc_mgr() { return children_arc_mgr_; }

  inline NodeMgr<Node>& mut_regst_desc_mgr() { return regst_desc_mgr_; }

  inline const ArcMgr& produced_regst_desc_mgr() const {
    return produced_regst_desc_mgr_;
  }
  inline ArcMgr& mut_produced_regst_desc_mgr() {
    return produced_regst_desc_mgr_;
  }

  inline const ArcMgr& subscribed_regst_desc_mgr() const {
    return subscribed_regst_desc_mgr_;
  }
  inline ArcMgr& mut_subscribed_regst_desc_mgr() {
    return subscribed_regst_desc_mgr_;
  }

 private:
  Node* source_;
  Node* sink_;
  NodeMgr<Node> node_mgr_;
  NodeMgr<Node> fake_node_mgr_;
  ArcMgr arc_mgr_;
  ArcMgr loss_arc_mgr_;
  ArcMgr children_arc_mgr_;
  ArcMgr ascendent_arc_mgr_;
  NodeMgr<Node> regst_desc_mgr_;
  ArcMgr produced_regst_desc_mgr_;
  ArcMgr subscribed_regst_desc_mgr_;
  NodeMgr<DeviceNode> device_mgr_;
  HasOneArcMgr device_arc_mgr_;
};

class RegstDesc : public Node {
 public:
  RegstDesc(const std::string name) : Node(name) {}
  RegstDesc() : Node() {}
  virtual ~RegstDesc() {}

  uint64_t regst_size() const { return regst_size_; }
  uint64_t& mut_regst_size() { return regst_size_; }

 private:
  uint64_t regst_size_ = 1u;
};

typedef Node Regst;

class Mode;

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_NODE_H_
