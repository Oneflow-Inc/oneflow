#include "oneflow/xrt/graph/node_attr.h"

namespace oneflow {
namespace xrt {

template <>
Shape GetNodeAttr<Shape>(const XrtNode *node, const std::string &attr_name) {
  DCHECK(HasFieldInPbMessage(node->param(), attr_name));
  return Shape(GetValFromPbMessage<ShapeProto>(node->param(), attr_name));
}

template <>
PbMessagePtr GetNodeAttr<PbMessagePtr>(const XrtNode *node,
                                       const std::string &attr_name) {
  DCHECK(HasFieldInPbMessage(node->param(), attr_name));
  return const_cast<PbMessagePtr>(
      &GetMessageInPbMessage(node->param(), attr_name));
}

std::string GetNodeAttrAsString(const XrtNode *node,
                                const std::string &attr_name) {
  return GetNodeAttr<std::string>(node, attr_name);
}

}  // namespace xrt
}  // namespace oneflow
