#include "oneflow/core/schedule/sxml.h"
namespace oneflow {
namespace schedule {

void SXMLNode::ForEachChild(const std::function<void(const SXML&)>& cb) const {
  for (const auto& child : children()) {
    if (child.node_type() == SXMLNode::kGroupNode) {
      child.node().ForEachChild(cb);
    } else {
      cb(child);
    }
  }
}

void SXMLNode::InitChildren(std::list<SXML>&& children) {
  for (SXML& node : children) { children_.push_back(std::move(node)); }
}

template<>
std::string SXMLNode::ToString<SXMLNode::kPlainNode>(int depth) const {
  return value();
}

template<>
std::string SXMLNode::ToString<SXMLNode::kGroupNode>(int depth) const {
  UNEXPECTED_RUN();
  return "";
}

template<>
std::string SXMLNode::ToString<SXMLNode::kKeyValueNode>(int depth) const {
  return "<" + tag_name() + ">" + value() + "</" + tag_name() + ">";
}

template<>
std::string SXMLNode::ToString<SXMLNode::kAttributesNode>(int depth) const {
  CHECK(tag_name() == "@");
  std::string attributes;
  ForEachChild([&](const SXML& child) {
    attributes += " " + child.field() + "=\"" + child.value() + "\"";
  });
  return attributes;
}

template<>
std::string SXMLNode::ToString<SXMLNode::kTagNode>(int depth) const {
  std::string attributes;
  std::string content;
  ForEachChild([&](const SXML& child) {
    if (child.node_type() == SXMLNode::kAttributesNode) {
      attributes += child.ToString(depth);
    } else {
      content += "\n" + std::string(depth + 1, ' ') + child.ToString(depth + 1);
    }
  });
  return "<" + tag_name() + attributes + ">" + content
         + (content.size() ? "\n" + std::string(depth, ' ') : "") + "</"
         + tag_name() + ">";
}

std::string SXML::ToString(int depth) const {
  switch (node_->node_type()) {
#define SXML_TO_STRING_ENTRY(node_type, index) \
  case SXMLNode::node_type: return node_->ToString<SXMLNode::node_type>(depth);
    OF_PP_FOR_EACH_TUPLE(SXML_TO_STRING_ENTRY, SXML_NODE_TYPE_SEQ)
    default: UNEXPECTED_RUN();
  }
  return "";
}

}  // namespace schedule
}  // namespace oneflow
