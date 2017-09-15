#ifndef ONEFLOW_CORE_SCHEDULE_SNODE_H_
#define ONEFLOW_CORE_SCHEDULE_SNODE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {
namespace schedule {

// static schedule node
class SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SNode);
  explicit SNode(const std::string name) : name_(name) {}
  SNode() = default;
  virtual ~SNode() = default;

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

  inline uint64_t GetAutoIncrementId() {
    static uint64_t counter = 1;
    counter++;
    return counter;
  }

  template<typename... Args>
  NodeType* Create(Args&&... args) {
    std::unique_ptr<NodeType> node(new NodeType(std::forward<Args>(args)...));
    NodeType* ret = node.get();
    do { node->mut_id() = GetAutoIncrementId(); } while (Find(node->id()));
    if (!Insert(std::move(node))) { ret = nullptr; }
    return ret;
  }

  template<typename... Args>
  NodeType* CreateWithId(uint64_t id, Args&&... args) {
    if (Find(id)) { return nullptr; }
    std::unique_ptr<NodeType> node(new NodeType(std::forward<Args>(args)...));
    node->mut_id() = id;
    NodeType* ret = node.get();
    if (!Insert(std::move(node))) { ret = nullptr; }
    return ret;
  }

  template<typename... Args>
  NodeType* CreateIfNotFound(const std::string& name, Args&&... args) {
    NodeType* node = Find(name);
    return node ? node : Create(std::forward<Args>(args)...);
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
               const std::function<void(const NodeType&)>& cb) const {
    return Find(name, [&](NodeType* node) { cb(*node); });
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

  void ForEach(const std::function<void(const NodeType&)>& cb) const {
    ForEach([&](NodeType* node) { cb(*node); });
  }
  void ForEach(const std::function<void(NodeType*)>& cb) const {
    for (const auto& pair : id2node_) { cb(pair.second.get()); }
  }

  NodeType* Find(uint64_t id) const {
    auto ret_itt = id2node_.find(id);
    if (id2node_.end() == ret_itt) { return nullptr; }
    return ret_itt->second.get();
  }

  void Delete(uint64_t id) {
    NodeType* node = Find(id);
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
  Arc(SrcNodeType* src_node, DstNodeType* dst_node) {
    Init(src_node, dst_node, "");
  }

  Arc(SrcNodeType* src_node, DstNodeType* dst_node, const std::string& name) {
    Init(src_node, dst_node, name);
  }

  void Init(SrcNodeType* src, DstNodeType* dst, const std::string& name) {
    src_node_ = src;
    dst_node_ = dst;
    mut_name() = name;
  }

  typedef SrcNodeType src_node_type;
  typedef DstNodeType dst_node_type;

  //	getter
  SrcNodeType* src_node() const { return src_node_; }
  DstNodeType* dst_node() const { return dst_node_; }
  std::string name() const {
    return "[" + src_node()->name() + "]->[" + dst_node()->name() + "]";
  }

 private:
  SrcNodeType* src_node_;
  DstNodeType* dst_node_;
};

template<typename ArcType,
         typename SrcNodeType = typename ArcType::src_node_type,
         typename DstNodeType = typename ArcType::dst_node_type>
class ArcMgr {
 public:
  ArcMgr() = default;
  virtual ~ArcMgr() = default;

  inline uint64_t GetAutoIncrementId() {
    static uint64_t counter = 0;
    counter++;
    return counter;
  }

  template<typename... Args>
  ArcType* CreateIfNotFound(SrcNodeType* src_node, DstNodeType* dst_node,
                            Args&&... args) {
    ArcType* node = Find(src_node, dst_node);
    return node ? node
                : Create(src_node, dst_node, std::forward<Args>(args)...);
  }

  int32_t Insert(std::unique_ptr<ArcType>&& arcp) {
    if (Find(arcp->src_node(), arcp->dst_node())) { return 0; }
    from2to2arc_[arcp->src_node()][arcp->dst_node()] = arcp.get();
    to2from2arc_[arcp->dst_node()][arcp->src_node()] = arcp.get();
    id2arc_[arcp->id()] = std::move(arcp);
    return 1;
  }

  uint32_t Input(const DstNodeType* dst_node) const {
    return Input(dst_node, [](SrcNodeType* src_node) {});
  }

  uint32_t Input(const DstNodeType* dst_node,
                 std::list<SrcNodeType*>* l) const {
    return Input(dst_node,
                 [l](SrcNodeType* src_node) { l->push_back(src_node); });
  }

  uint32_t Input(const DstNodeType* dst_node,
                 const std::function<void(const SrcNodeType&)>& cb) const {
    return Input(dst_node, [&](SrcNodeType* node) { cb(*node); });
  }
  uint32_t Input(const DstNodeType* dst_node,
                 const std::function<void(SrcNodeType*)>& cb) const {
    return InputArc(dst_node, [&cb](ArcType* arcp) { cb(arcp->src_node()); });
  }

  uint32_t Input(const DstNodeType* dst_node, SrcNodeType** src_node) const {
    return InputArc(dst_node,
                    [src_node](ArcType* p) { *src_node = p->src_node(); });
  }

  uint32_t Input(const std::list<DstNodeType*>& dst_nodes) const {
    return Input(dst_nodes, [](SrcNodeType*) {});
  }

  uint32_t Input(const std::list<DstNodeType*>& dst_nodes,
                 const std::function<void(const SrcNodeType&)>& cb) const {
    return Input(dst_nodes, [&](SrcNodeType* node) { cb(*node); });
  }
  uint32_t Input(const std::list<DstNodeType*>& dst_nodes,
                 const std::function<void(SrcNodeType*)>& cb) const {
    std::unordered_set<SrcNodeType*> src_nodes;
    InputArc(dst_nodes,
             [&](ArcType* arc) { src_nodes.insert(arc->src_node()); });
    for (auto node : src_nodes) { cb(node); }
    return src_nodes.size();
  }

  uint32_t InputArc(const DstNodeType* dst_node, std::list<ArcType*>* l) const {
    return InputArc(dst_node, [l](ArcType* arc) { l->push_back(arc); });
  }

  uint32_t InputArc(const std::list<DstNodeType*>& to_nodes,
                    const std::function<void(const ArcType&)>& cb) const {
    return InputArc(to_nodes, [&](ArcType* arc) { cb(*arc); });
  }

  uint32_t InputArc(const std::list<DstNodeType*>& to_nodes,
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

  uint32_t InputArc(const DstNodeType* dst_node, ArcType* ptr) const {
    return InputArc(dst_node, [&ptr](ArcType* p) { ptr = p; });
  }

  uint32_t InputArc(const DstNodeType* dst_node,
                    const std::function<void(const ArcType&)>& cb) const {
    return InputArc(dst_node, [&](ArcType* arc) { cb(*arc); });
  }
  uint32_t InputArc(const DstNodeType* dst_node,
                    const std::function<void(ArcType*)>& cb) const {
    uint32_t count = 0;
    auto itt = to2from2arc_.find(const_cast<DstNodeType*>(dst_node));
    if (itt == to2from2arc_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  uint32_t Output(const SrcNodeType* src_node, DstNodeType** dst_node) const {
    return OutputArc(src_node,
                     [&dst_node](ArcType* p) { *dst_node = p->dst_node(); });
  }

  uint32_t Output(const SrcNodeType* src_node) const {
    return Output(src_node, [](DstNodeType* dst_node) {});
  }

  uint32_t Output(const SrcNodeType* src_node,
                  std::list<DstNodeType*>* l) const {
    return Output(src_node,
                  [&l](DstNodeType* dst_node) { l->push_back(dst_node); });
  }

  uint32_t Output(const SrcNodeType* src_node,
                  const std::function<void(const DstNodeType&)>& cb) const {
    return Output(src_node, [&cb](DstNodeType* ptr) { cb(*ptr); });
  }

  uint32_t Output(const SrcNodeType* src_node,
                  const std::function<void(DstNodeType*)>& cb) const {
    return OutputArc(src_node, [&cb](ArcType* arcp) { cb(arcp->dst_node()); });
  }

  uint32_t Output(const std::list<SrcNodeType*>& src_nodes) const {
    return Output(src_nodes, [](DstNodeType*) {});
  }

  uint32_t Output(const std::list<SrcNodeType*>& src_nodes,
                  const std::function<void(const DstNodeType&)>& cb) const {
    return Output(src_nodes, [&cb](DstNodeType* ptr) { cb(*ptr); });
  }

  uint32_t Output(const std::list<SrcNodeType*>& src_nodes,
                  const std::function<void(DstNodeType*)>& cb) const {
    std::unordered_set<DstNodeType*> dst_nodes;
    OutputArc(src_nodes,
              [&](ArcType* arc) { dst_nodes.insert(arc->dst_node()); });
    for (auto node : dst_nodes) { cb(node); }
    return dst_nodes.size();
  }

  uint32_t OutputArc(const SrcNodeType* src_node, ArcType* ptr) const {
    return OutputArc(src_node, [&ptr](ArcType* p) { ptr = p; });
  }

  uint32_t OutputArc(const SrcNodeType* src_node,
                     std::list<ArcType*>* l) const {
    return OutputArc(src_node, [&l](ArcType* arc) { l->push_back(arc); });
  }

  uint32_t OutputArc(const SrcNodeType* src_node,
                     std::function<void(const ArcType&)> cb) const {
    return OutputArc(src_node, [&](ArcType* arc) { cb(*arc); });
  }
  uint32_t OutputArc(const SrcNodeType* src_node,
                     std::function<void(ArcType*)> cb) const {
    uint32_t count = 0;
    auto itt = from2to2arc_.find(const_cast<SrcNodeType*>(src_node));
    if (itt == from2to2arc_.end()) { return count; }
    for (auto jtt = itt->second.begin(); jtt != itt->second.end(); jtt++) {
      cb(jtt->second);
      count++;
    }
    return count;
  }

  uint32_t OutputArc(const std::list<SrcNodeType*>& from_nodes,
                     const std::function<void(const ArcType&)>& cb) const {
    return OutputArc(from_nodes, [&](ArcType* arc) { cb(*arc); });
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
            const std::function<void(const ArcType&)>& cb) const {
    Find(from_nodes, to_nodes, [&](ArcType* arc) { cb(*arc); });
  }

  void Find(const std::list<SrcNodeType*>& from_nodes,
            const std::list<DstNodeType*>& to_nodes,
            const std::function<void(ArcType*)>& cb) const {
    for (SrcNodeType* src_node : from_nodes) {
      auto itt = from2to2arc_.find(src_node);
      if (itt == from2to2arc_.end()) { continue; }
      for (DstNodeType* dst_node : to_nodes) {
        auto jtt = itt->second.find(dst_node);
        if (jtt == itt->second.end()) { continue; }
        cb(jtt->second);
      }
    }
  }

  ArcType* Find(SrcNodeType* src_node, DstNodeType* dst_node) const {
    auto itt = from2to2arc_.find(src_node);
    if (itt == from2to2arc_.end()) { return nullptr; }
    auto jtt = itt->second.find(dst_node);
    if (jtt == itt->second.end()) { return nullptr; }
    return jtt->second;
  }

  ArcType* Find(uint64_t id) const {
    auto itt = id2arc_.find(id);
    if (itt == id2arc_.end()) { return nullptr; }
    return itt->second.get();
  }

  void Delete(uint64_t id) {
    ArcType* arcp = Find(id);
    if (!arcp) { return; }
    from2to2arc_[arcp->src_node()].erase(arcp->dst_node());
    if (!from2to2arc_[arcp->src_node()].size()) {
      from2to2arc_.erase(arcp->src_node());
    }
    to2from2arc_[arcp->dst_node()].erase(arcp->src_node());
    if (!to2from2arc_[arcp->dst_node()].size()) {
      to2from2arc_.erase(arcp->dst_node());
    }
    id2arc_.erase(id);
  }

  void Delete(SrcNodeType* src_node, DstNodeType* dst_node) {
    ArcType* arcp = Find(src_node, dst_node);
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
    std::unique_ptr<ArcType> p(new ArcType(std::forward<Args>(args)...));
    ArcType* ret = p.get();
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
  ArcType* CreateIfNotFound(SrcNodeType* src_node, DstNodeType* dst_node,
                            Args&&... args) {
    std::list<DstNodeType*> dst_nodes;
    this->Output(src_node, &dst_nodes);

    for (DstNodeType* node : dst_nodes) { this->Delete(src_node, node); }
    return ArcMgr<ArcType>::CreateIfNotFound(src_node, dst_node,
                                             std::forward<Args>(args)...);
  }
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SNODE_H_
