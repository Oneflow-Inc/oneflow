#ifndef ONEFLOW_CORE_SCHEDULE_SXML_H_
#define ONEFLOW_CORE_SCHEDULE_SXML_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace schedule {

class SXML;

class SXMLNode final {
#define SXML_NODE_TYPE_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(kTagNode, 0)        \
  OF_PP_MAKE_TUPLE_SEQ(kAttributesNode, 1) \
  OF_PP_MAKE_TUPLE_SEQ(kKeyValueNode, 2)   \
  OF_PP_MAKE_TUPLE_SEQ(kPlainNode, 3)

 public:
  enum NodeType {
#define DECLARE_SXML_NODE_TYPE(type, value) type = value,
    OF_PP_FOR_EACH_TUPLE(DECLARE_SXML_NODE_TYPE, SXML_NODE_TYPE_SEQ)
  };
  SXMLNode(const std::string& tag) : node_type_(kTagNode), key_(tag) {}
  SXMLNode(const std::string& key, const std::string& value)
      : node_type_(EvalNodeType(key)), key_(key), value_(value) {}
  SXMLNode(const std::string& key, std::list<SXML>&& children)
      : node_type_(EvalNodeType(key)), key_(key) {
    InitChildren(std::move(children));
  }

  ~SXMLNode() = default;

  INLINE NodeType node_type() const { return node_type_; }
  INLINE const std::string& tag_name() const { return key_; }
  INLINE const std::string& field() const { return key_; }
  INLINE const std::string& value() const { return value_; }
  INLINE const std::list<SXML>& children() const { return children_; }

  template<NodeType node_type>
  std::string ToString(int depth) const;

 private:
  void InitChildren(std::list<SXML>&& children);
  static NodeType EvalNodeType(const std::string& tag_name) {
    if (tag_name == "") return kPlainNode;
    if (tag_name == "@") return kAttributesNode;
    return kTagNode;
  }
  NodeType node_type_;
  std::string key_;
  std::string value_;
  std::list<SXML> children_;
};

class SXML final {
 public:
  typedef SXMLNode::NodeType NodeType;
  explicit SXML(const SXML& node)
      : node_(of_make_unique<SXMLNode>(node.node())) {}
  explicit SXML(SXML&& node) : node_(node.move_node()) {}
  explicit SXML(std::unique_ptr<SXMLNode>&& node) : node_(std::move(node)) {}
  explicit SXML(const std::string& content)
      : node_(of_make_unique<SXMLNode>(content)) {}
  SXML(const std::string& key, const std::string& value)
      : node_(of_make_unique<SXMLNode>(key, value)) {}
  SXML(const std::string& key, std::list<SXML>&& children)
      : node_(of_make_unique<SXMLNode>(key, std::move(children))) {}

  INLINE std::unique_ptr<SXMLNode> move_node() { return std::move(node_); }

  std::string ToString(int dpeth) const;
  std::string ToString() const { return ToString(0); }

  INLINE NodeType node_type() const { return node_->node_type(); }
  INLINE const std::string& tag_name() const { return node_->tag_name(); }
  INLINE const std::string& field() const { return node_->field(); }
  INLINE const std::string& value() const { return node_->value(); }
  INLINE const std::list<SXML>& children() const { return node_->children(); }

 private:
  INLINE const SXMLNode& node() const { return *node_; }
  std::unique_ptr<SXMLNode> node_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SXML_H_
