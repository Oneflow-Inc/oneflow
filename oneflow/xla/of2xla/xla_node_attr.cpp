#include "oneflow/xla/of2xla/xla_node_attr.h"

namespace oneflow {
namespace mola {

template <>
Shape GetNodeAttr<Shape>(const XlaNode *node, const std::string &attr_name) {
  DCHECK(HasFieldInPbMessage(node->proto_conf(), attr_name));
  return Shape(GetValFromPbMessage<ShapeProto>(node->proto_conf(), attr_name));
}

template <>
PbMessagePtr GetNodeAttr<PbMessagePtr>(const XlaNode *node,
                                       const std::string &attr_name) {
  DCHECK(HasFieldInPbMessage(node->proto_conf(), attr_name));
  return const_cast<PbMessagePtr>(
      &GetMessageInPbMessage(node->proto_conf(), attr_name));
}

std::string GetNodeAttrAsString(const XlaNode *node,
                                const std::string &attr_name) {
  return GetNodeAttr<std::string>(node, attr_name);
}

}  // namespace mola
}  // namespace oneflow
