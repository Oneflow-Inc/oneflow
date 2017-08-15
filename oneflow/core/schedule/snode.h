/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_SNODE_H_
#define ONEFLOW_CORE_SCHEDULE_SNODE_H_

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

  int32_t Insert(std::unique_ptr<NodeType>&& node) {
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

  int32_t Find(const std::string& name,
               const std::function<void(NodeType*)>& cb) const {
    auto itt = name2id2node_.find(name);
    int32_t count = 0;
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

  int32_t Insert(std::unique_ptr<ArcType>&& arcp) {
    if (Find(arcp->from(), arcp->to())) { return 0; }
    from2to2arc_[arcp->from()][arcp->to()] = arcp.get();
    to2from2arc_[arcp->to()][arcp->from()] = arcp.get();
    id2arc_[arcp->id()] = std::move(arcp);
    return 1;
  }

  uint32_t Input(const DstNodeType* to) const {
    return Input(to, [](SrcNodeType* from) {});
  }

  uint32_t Input(const DstNodeType* to, std::list<SrcNodeType*>* l) const {
    return Input(to, [l](SrcNodeType* from) { l->push_back(from); });
  }

  uint32_t Input(const DstNodeType* to,
                 const std::function<void(SrcNodeType*)>& cb) const {
    return InputArc(to, [&cb](ArcType* arcp) { cb(arcp->from()); });
  }

  uint32_t Input(const DstNodeType* to, SrcNodeType** from) const {
    return InputArc(to, [from](ArcType* p) { *from = p->from(); });
  }

  uint32_t InputArc(const DstNodeType* to, std::list<ArcType*>* l) const {
    return InputArc(to, [l](ArcType* arc) { l->push_back(arc); });
  }

  uint32_t InputArc(std::list<DstNodeType*> to_nodes,
                    const std::function<void(ArcType*)>& cb) const {
    uint32_t count = 0;
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

  uint32_t InputArc(const DstNodeType* to, ArcType* ptr) const {
    return InputArc(to, [&ptr](ArcType* p) { ptr = p; });
  }

  uint32_t InputArc(const DstNodeType* to,
                    const std::function<void(ArcType*)>& cb) const {
    uint32_t count = 0;
    auto itt = to2from2arc_.find(const_cast<DstNodeType*>(to));
    if (itt == to2from2arc_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  uint32_t Output(const SrcNodeType* from, DstNodeType** to) const {
    return OutputArc(from, [&to](ArcType* p) { *to = p->to(); });
  }

  uint32_t Output(const SrcNodeType* from) const {
    return Output(from, [](DstNodeType* to) {});
  }

  uint32_t Output(const SrcNodeType* from, std::list<DstNodeType*>* l) const {
    return Output(from, [&l](DstNodeType* to) { l->push_back(to); });
  }

  uint32_t Output(const SrcNodeType* from,
                  const std::function<void(DstNodeType*)>& cb) const {
    return OutputArc(from, [&cb](ArcType* arcp) { cb(arcp->to()); });
  }

  uint32_t OutputArc(const SrcNodeType* from, ArcType* ptr) const {
    return OutputArc(from, [&ptr](ArcType* p) { ptr = p; });
  }

  uint32_t OutputArc(const SrcNodeType* from, std::list<ArcType*>* l) const {
    return OutputArc(from, [&l](ArcType* arc) { l->push_back(arc); });
  }

  uint32_t OutputArc(const SrcNodeType* from,
                     std::function<void(ArcType*)> cb) const {
    uint32_t count = 0;
    auto itt = from2to2arc_.find(const_cast<SrcNodeType*>(from));
    if (itt == from2to2arc_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  uint32_t OutputArc(const std::list<SrcNodeType*>& from_nodes,
                     const std::function<void(ArcType*)>& cb) const {
    uint32_t count = 0;
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

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SNODE_H_
