#ifndef ONEFLOW_XRT_XRT_NODE_ATTR_H_
#define ONEFLOW_XRT_XRT_NODE_ATTR_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/xrt/graph/node.h"

namespace oneflow {
namespace xrt {

template <typename T>
inline T GetNodeAttr(const XrtNode *node, const std::string &attr_name) {
  DCHECK(HasFieldInPbMessage(node->param(), attr_name));
  return GetValFromPbMessage<T>(node->param(), attr_name);
}

template <>
Shape GetNodeAttr<Shape>(const XrtNode *node, const std::string &attr_name);

typedef oneflow::PbMessage *PbMessagePtr;
template <>
PbMessagePtr GetNodeAttr<PbMessagePtr>(const XrtNode *node,
                                       const std::string &attr_name);

std::string GetNodeAttrAsString(const XrtNode *node,
                                const std::string &attr_name);

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XRT_NODE_ATTR_H_
