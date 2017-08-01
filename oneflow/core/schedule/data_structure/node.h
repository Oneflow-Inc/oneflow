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

#include "oneflow/core/schedule/utils/utils.h"

namespace oneflow {
namespace schedule {

template<typename T, typename... Args>
inline std::unique_ptr<T> unique_ptr_new(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

inline uint64_t GetAutoIncrementId() {
  static uint64_t counter = 0;
  counter++;
  return counter;
}

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

  void Walk(const std::function<void(Node*)>& cb);
  void WalkArc(const std::function<void(Arc*)>& cb);
  uint32_t DeviceCount();
  uint32_t Depth();
  void WalkReverse(const std::function<void(Node*)>& cb);
  void WalkArcReverse(const std::function<void(Arc*)>& cb);
  void ForeachArc(const std::function<void(Arc*)>& cb) const;
  void UpdateSourceAndSink();
  int LossNodes(std::list<Node*>* l) const;
  Node* source() const { return source_; }
  Node*& mut_source() { return source_; }

  Node* sink() const { return sink_; }
  Node*& mut_sink() { return sink_; }

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

template<typename K, typename C, typename E = typename C::value_type,
         typename F = std::function<K(const E&)>>
std::unique_ptr<std::unordered_map<K, std::list<E>>> XGroupBy(
    const C& container, const F& f) {
  auto collect = unique_ptr_new<std::unordered_map<K, std::list<E>>>();
  for (const E& elem : container) { (*collect)[f(elem)].push_back(elem); }
  return collect;
}

template<typename NV, typename C, typename K = typename C::key_type,
         typename V = typename C::mapped_type,
         typename F = std::function<NV(const V&)>>
std::unique_ptr<std::unordered_map<K, NV>> XAssocVMap(const C& container,
                                                      const F& f) {
  auto collect = unique_ptr_new<std::unordered_map<K, NV>>();
  for (const auto& p : container) { (*collect)[p.first] = f(p.second); }
  return collect;
}

template<typename C, typename T = typename C::const_iterator>
T XAssocKMin(const C& container) {
  auto itt = container.begin();

  if (itt != container.end()) {
    auto jtt = itt;
    for (jtt++; jtt != container.end(); jtt++) {
      if (jtt->first < itt->first) { itt = jtt; }
    }
  }

  return itt;
}

template<typename E, typename C, typename F = std::function<bool(const E&)>>
std::unique_ptr<std::list<E>> XFilter(const C& container, const F& f) {
  auto collect = unique_ptr_new<std::list<E>>();
  for (const E& elem : container) {
    if (f(elem)) { collect->push_back(elem); }
  }
  return collect;
}

template<typename K, typename C,
         typename E = std::pair<typename C::key_type, typename C::mapped_type>,
         typename F = std::function<K(const E&)>>
std::unique_ptr<std::unordered_set<K>> XAssocDistinct(const C& container,
                                                      const F& f) {
  auto collect = unique_ptr_new<std::unordered_set<K>>();
  for (const E& elem : container) { collect->insert(f(elem)); }
  return collect;
}

template<typename K, typename C, typename E = typename C::value_type,
         typename F = std::function<K(const E&)>>
std::unique_ptr<std::unordered_set<K>> XDistinct(const C& container,
                                                 const F& f) {
  auto collect = unique_ptr_new<std::unordered_set<K>>();
  for (const E& elem : container) { collect->insert(f(elem)); }
  return collect;
}

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

class Session;

class Mode;

class SessionLogger {
 public:
  SessionLogger() {}
  void Clear();
  void UpdateTimeGapToLoss(Session* session, Mode* strategy);
  void UpdateDuration(Session* session, Mode* strategy);
  void UpdateInterval(Session* session, Mode* strategy);
  void MergeTimeGapToLossInPlace(SessionLogger* logger);
  float GetDurationByTimeGapToLoss(Arc* from, Arc* to);

  std::unordered_map<Arc*, std::pair<int32_t, int32_t>> arc2ended_at_;
  std::unordered_map<Arc*, std::unordered_map<Node*, float>>
      start_time_gap_to_loss_;
  std::unordered_map<Arc*, std::unordered_map<Node*, float>>
      end_time_gap_to_loss_;
  std::unordered_map<Node*, int32_t> device2ended_at_;
  std::unordered_map<Node*, float> node2interval_;
  float max_interval_ = 0.0;
  std::unordered_map<Node*, float> regst_desc2duration_;
};

class Session {
 public:
  explicit Session(GraphNode* root, uint32_t nr_batch = 2u)
      : root_(root), logger_(unique_ptr_new<SessionLogger>()) {
    mut_root()->UpdateSourceAndSink();
    mut_root()->InitDepth();
    mut_root()->InitAscendentArc();
    auto nr_device = root->DeviceCount();
    auto depth = root->Depth();
    nr_base_batch_ = std::min(nr_device, depth);
    nr_batch_ = std::max(nr_batch, nr_device * 3);
    NewBatchs();
  }

  struct PipeSpec {
    float duration;
    float freq;
    uint32_t count;
  };
  typedef std::unordered_map<uint32_t, PipeSpec> PipeCount;
  typedef std::unordered_map<Node*, std::pair<int32_t, int32_t>> NodeEndTime;
  typedef std::unordered_map<Node*, uint32_t> NodeAscMaxTime;
  typedef std::function<std::unique_ptr<std::unordered_map<Node*, Arc*>>(
      std::unordered_set<Arc*>* tokens)>
      PickStrategy;

  Node* GetInstanceDevice(Arc* instance);

  void ForeachRegstDesc(const std::function<void(Node*)>& cb) const;
  void NewSourceTokens();
  void NewSinkTokens();
  void ClearTmpData();
  void NewBatchs();
  void InitNodeBatchInstance(Node* node);
  std::unique_ptr<std::list<Node*>> GetBatchNodes();
  std::unique_ptr<PipeCount> RegstDescCount(bool bottleneck = true);

  inline const GraphNode* root() const { return root_; }
  inline GraphNode* mut_root() { return root_; }
  SessionLogger* logger() { return logger_.get(); }
  std::unique_ptr<SessionLogger>& mut_logger() { return logger_; }
  std::unique_ptr<SessionLogger> GetLoggerThenReset() {
    auto ret = std::move(logger_);
    logger_ = unique_ptr_new<SessionLogger>();
    return ret;
  }

  GraphNode* root_;
  uint32_t nr_batch_;
  uint32_t nr_base_batch_;
  std::unordered_set<Arc*> tokens_;
  std::unique_ptr<SessionLogger> logger_;

  inline const NodeMgr<Node>& batch_node_mgr() const { return batch_node_mgr_; }
  inline NodeMgr<Node>& mut_batch_node_mgr() { return batch_node_mgr_; }

  inline const ArcMgr& batch_arc_mgr() const { return batch_arc_mgr_; }
  inline ArcMgr& mut_batch_arc_mgr() { return batch_arc_mgr_; }

  inline const NodeMgr<Node>& batch_instance_node_mgr() const {
    return batch_instance_node_mgr_;
  }
  inline NodeMgr<Node>& mut_batch_instance_node_mgr() {
    return batch_instance_node_mgr_;
  }

 private:
  NodeMgr<Node> batch_instance_node_mgr_;
  NodeMgr<Node> batch_node_mgr_;
  ArcMgr batch_arc_mgr_;
};

class Strategy {
 public:
  Strategy(Session* sess) : sess_(sess) {}
  virtual ~Strategy() {}

  inline Session* Sess() { return sess_; }

 protected:
  Session* sess_;
};

class DirectionStrategy : public Strategy {
 public:
  DirectionStrategy(Session* sess) : Strategy(sess) {}
  virtual ~DirectionStrategy() {}

  virtual int32_t GetTime(int32_t x) = 0;
  virtual int32_t GetStartTime(const std::pair<int32_t, int32_t>& p) = 0;
  virtual int32_t GetEndTime(const std::pair<int32_t, int32_t>& p) = 0;
  virtual void NewStartTokens() = 0;
  virtual unsigned int NextArc(Node* node,
                               const std::function<void(Arc*)>& cb) = 0;
  virtual unsigned int Next(Node* node,
                            const std::function<void(Node*)>& cb) = 0;
  virtual unsigned int PrevArc(Node* node,
                               const std::function<void(Arc*)>& cb) = 0;
  virtual unsigned int Prev(Node* node,
                            const std::function<void(Node*)>& cb) = 0;
  virtual Arc* GetNextNodeInstance(Arc* arc) = 0;
  virtual bool CompareInstanceOrder(Arc* instance_a, Arc* instance_b) = 0;
  virtual Arc* PickInstanceToRun(const std::list<Arc*>& instances);
  virtual int HoldingRegstDesc(Node* node,
                               const std::function<void(Node*)>& cb) = 0;
  virtual int RegstDescReleasingNode(Node* regst_desc,
                                     const std::function<void(Node*)>& cb) = 0;
  virtual Node* StartNode() = 0;
  virtual Node* EndNode() = 0;
  virtual Node* EndBatch() = 0;
  virtual uint32_t NextBatchId(uint32_t batch_id) = 0;
  virtual Node* GetFrom(Arc* arc) = 0;
  virtual Node* GetTo(Arc* arc) = 0;
};

class PositiveStrategy : public DirectionStrategy {
 public:
  PositiveStrategy(Session* sess) : DirectionStrategy(sess) {}
  virtual ~PositiveStrategy() {}
  int32_t GetTime(int32_t x) { return x; }
  int32_t GetStartTime(const std::pair<int32_t, int32_t>& p) {
    return GetTime(p.first);
  }
  int32_t GetEndTime(const std::pair<int32_t, int32_t>& p) {
    return GetTime(p.second);
  }
  void NewStartTokens();
  unsigned int NextArc(Node* node, const std::function<void(Arc*)>& cb);
  unsigned int Next(Node* node, const std::function<void(Node*)>& cb);
  unsigned int PrevArc(Node* node, const std::function<void(Arc*)>& cb);
  unsigned int Prev(Node* node, const std::function<void(Node*)>& cb);
  Arc* GetNextNodeInstance(Arc* arc);
  bool CompareInstanceOrder(Arc* instance_a, Arc* instance_b);
  int HoldingRegstDesc(Node* node, const std::function<void(Node*)>& cb);
  int RegstDescReleasingNode(Node* regst_desc,
                             const std::function<void(Node*)>& cb);
  Node* StartNode() { return Sess()->root()->source(); }
  Node* EndNode() { return Sess()->root()->sink(); }
  Node* EndBatch() {
    return Sess()->batch_node_mgr().Find(Sess()->nr_batch_ - 1);
  }
  Node* GetFrom(Arc* arc) { return arc->from(); }
  Node* GetTo(Arc* arc) { return arc->to(); }
  uint32_t NextBatchId(uint32_t batch_id) { return batch_id + 1; }
};

class NegativeStrategy : public DirectionStrategy {
 public:
  NegativeStrategy(Session* sess) : DirectionStrategy(sess) {}
  virtual ~NegativeStrategy() {}
  virtual int32_t GetTime(int32_t x) { return -x; }
  virtual int32_t GetStartTime(const std::pair<int32_t, int32_t>& p) {
    return GetTime(p.second);
  }
  virtual int32_t GetEndTime(const std::pair<int32_t, int32_t>& p) {
    return GetTime(p.first);
  }
  void NewStartTokens();
  unsigned int NextArc(Node* node, const std::function<void(Arc*)>& cb);
  unsigned int Next(Node* node, const std::function<void(Node*)>& cb);
  unsigned int PrevArc(Node* node, const std::function<void(Arc*)>& cb);
  unsigned int Prev(Node* node, const std::function<void(Node*)>& cb);
  Arc* GetNextNodeInstance(Arc* arc);
  bool CompareInstanceOrder(Arc* instance_a, Arc* instance_b);
  int HoldingRegstDesc(Node* node, const std::function<void(Node*)>& cb);
  int RegstDescReleasingNode(Node* regst_desc,
                             const std::function<void(Node*)>& cb);
  Node* StartNode() { return Sess()->root()->sink(); }
  Node* EndNode() { return Sess()->root()->source(); }
  Node* EndBatch() { return Sess()->batch_node_mgr().Find(0u); }
  Node* GetFrom(Arc* arc) { return arc->to(); }
  Node* GetTo(Arc* arc) { return arc->from(); }
  uint32_t NextBatchId(uint32_t batch_id) { return batch_id - 1; }
};

typedef Arc InstanceArc;

class EvaluationStrategy : public Strategy {
 public:
  EvaluationStrategy(DirectionStrategy* direction)
      : Strategy(direction->Sess()), direction_(direction) {}
  virtual int32_t GetAscendentEndedAt(Arc* instance);
  virtual void TimeLinePushBack(InstanceArc*, DeviceNode*) = 0;
  virtual void Retiming() = 0;

 protected:
  DirectionStrategy* direction_;
};

class EagerStrategy : public EvaluationStrategy {
 public:
  EagerStrategy(DirectionStrategy* direction) : EvaluationStrategy(direction) {}
  void TimeLinePushBack(InstanceArc* instance, DeviceNode* device) {}
  void Retiming(){};
};

class LazyStrategy : public EvaluationStrategy {
 public:
  LazyStrategy(DirectionStrategy* direction) : EvaluationStrategy(direction) {
    InitTimeNet();
  }

  void TimeLinePushBack(InstanceArc* instance, DeviceNode* device);
  void Retiming();

 protected:
  inline const ArcMgr& timenet_arc_mgr() const { return timenet_arc_mgr_; }
  inline ArcMgr& mut_timenet_arc_mgr() { return timenet_arc_mgr_; }
  void InitTimeNet();
  void WalkTimeNetReverse(const std::function<void(InstanceArc*)>& cb);
  ArcMgr timenet_arc_mgr_;
  std::unordered_map<DeviceNode*, InstanceArc*> dev2current_instance_;
};

class ResourceStrategy : public Strategy {
 public:
  ResourceStrategy(DirectionStrategy* direction, EvaluationStrategy* evaluation)
      : Strategy(direction->Sess()),
        evaluation_(evaluation),
        direction_(direction) {
    InitFuncs();
  }
  virtual ~ResourceStrategy() {}
  virtual std::unique_ptr<std::unordered_map<Node*, Arc*>> Pick(
      std::unordered_set<Arc*>* tokens);
  virtual void BeforeRun(Arc* instance) = 0;
  virtual void AfterRun(Arc* instance) = 0;
  virtual int32_t GetAscendentEndedAt(Arc* instance);

  std::function<int32_t(Arc*)> get_ascendent_ended_at_;

 protected:
  void InitFuncs();
  virtual bool IsInstanceReady(Arc* instance);

  std::function<Arc*(Arc*)> get_node_instance_;
  std::function<bool(Arc*)> is_instance_ready_;
  std::function<Node*(Arc*)> get_instance_device_;
  EvaluationStrategy* evaluation_;
  DirectionStrategy* direction_;
  std::function<Arc*(const std::list<Arc*>&)> pick_instance_to_run_;
};

class UnlimitedStrategy : public ResourceStrategy {
 public:
  UnlimitedStrategy(DirectionStrategy* direction, EvaluationStrategy* evalution)
      : ResourceStrategy(direction, evalution) {}
  virtual void BeforeRun(Arc* instance) {}
  virtual void AfterRun(Arc* instance) {}
};

class LimitedStrategy : public ResourceStrategy {
 public:
  LimitedStrategy(DirectionStrategy* direction, EvaluationStrategy* evaluation,
                  const Session::PipeCount& pipe_count)
      : ResourceStrategy(direction, evaluation) {
    InitRegst(pipe_count);
    InitFuncIsInstanceReady();
  }
  void BeforeRun(Arc* instance);
  void AfterRun(Arc* instance);

 private:
  inline const NodeMgr<Regst>& regst_node_mgr() const {
    return regst_node_mgr_;
  }
  inline NodeMgr<Regst>& mut_regst_node_mgr() { return regst_node_mgr_; }
  inline const ArcMgr& regst_arc_mgr() const { return regst_arc_mgr_; }
  inline ArcMgr& mut_regst_arc_mgr() { return regst_arc_mgr_; }
  inline const HasOneArcMgr& r2rd_arc_mgr() const { return r2rd_arc_mgr_; }
  inline HasOneArcMgr& mut_r2rd_arc_mgr() { return r2rd_arc_mgr_; }

  void InitRegst(const Session::PipeCount& pipe_count);
  void InitFuncIsInstanceReady();
  bool IsAllRegstDescReady(Arc* instance);
  bool IsRegstDescReady(Node* regst_desc, Node* batch);
  Regst* FindFreeRegst(Node* regst_desc, Node* batch);
  bool IsRegstFree(Regst* regst);
  int32_t RegstDescEndedAt(Arc* instance);
  std::unordered_map<Regst*, int32_t> regst2ended_at_;
  std::unordered_map<Arc*, Regst*> regst_desc_instance2regst_;
  NodeMgr<Regst> regst_node_mgr_;
  ArcMgr regst_arc_mgr_;
  HasOneArcMgr r2rd_arc_mgr_;
};

class Mode : public Strategy {
 public:
  Mode(Session* sess) : Strategy(sess) {}
  virtual ~Mode() {}
  DEFINE_PURE_VIRTUAL_TYPE();
  inline int32_t GetTime(int32_t x) { return direction_->GetTime(x); }
  inline int32_t GetStartTime(const std::pair<int32_t, int32_t>& p) {
    return direction_->GetStartTime(p);
  }
  inline int32_t GetEndTime(const std::pair<int32_t, int32_t>& p) {
    return direction_->GetEndTime(p);
  }
  void Run();

 protected:
  inline void NewStartTokens() { return direction_->NewStartTokens(); }
  inline unsigned int NextArc(Node* node, const std::function<void(Arc*)>& cb) {
    return direction_->NextArc(node, cb);
  }
  inline unsigned int PrevArc(Node* node, const std::function<void(Arc*)>& cb) {
    return direction_->PrevArc(node, cb);
  }
  inline std::unique_ptr<std::unordered_map<Node*, Arc*>> Pick(
      std::unordered_set<Arc*>* tokens) {
    return resource_->Pick(tokens);
  }
  inline void TimeLinePushBack(InstanceArc* instance, DeviceNode* dev) {
    return evaluation_->TimeLinePushBack(instance, dev);
  }
  inline void Retiming() { return evaluation_->Retiming(); }
  inline void BeforeRun(Arc* instance) {
    //    evaluation_->BeforeRun(instance);
    resource_->BeforeRun(instance);
  }
  inline void AfterRun(Arc* instance) {
    //    evaluation_->AfterRun(instance);
    resource_->AfterRun(instance);
  }
  inline int32_t GetAscendentEndedAt(Arc* instance) {
    return resource_->get_ascendent_ended_at_(instance);
  }
  void SetStrategies(std::unique_ptr<DirectionStrategy>&& direction,
                     std::unique_ptr<EvaluationStrategy>&& evaluation,
                     std::unique_ptr<ResourceStrategy>&& resource) {
    direction_ = std::move(direction);
    evaluation_ = std::move(evaluation);
    resource_ = std::move(resource);
  }
  std::unique_ptr<DirectionStrategy> direction_;
  std::unique_ptr<EvaluationStrategy> evaluation_;
  std::unique_ptr<ResourceStrategy> resource_;
};

template<typename DirectionStrategyType,
         typename EvaluationStrategyType = LazyStrategy>
class UnlimitedMode : public Mode {
 public:
  explicit UnlimitedMode(Session* sess) : Mode(sess) {
    auto direction = unique_ptr_new<DirectionStrategyType>(sess);
    auto evaluation = unique_ptr_new<EvaluationStrategyType>(&*direction);
    auto resource =
        unique_ptr_new<UnlimitedStrategy>(&*direction, &*evaluation);
    SetStrategies(std::move(direction), std::move(evaluation),
                  std::move(resource));
  }
  DEFINE_METHOD_TYPE();
};

template<typename DirectionStrategyType,
         typename EvaluationStrategyType = EagerStrategy>
class LimitedMode : public Mode {
 public:
  LimitedMode(Session* sess, const Session::PipeCount& pipe_count)
      : Mode(sess) {
    auto direction = unique_ptr_new<DirectionStrategyType>(sess);
    auto evaluation = unique_ptr_new<EvaluationStrategyType>(&*direction);
    auto resource =
        unique_ptr_new<LimitedStrategy>(&*direction, &*evaluation, pipe_count);
    SetStrategies(std::move(direction), std::move(evaluation),
                  std::move(resource));
  }
  DEFINE_METHOD_TYPE();
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_NODE_H_
