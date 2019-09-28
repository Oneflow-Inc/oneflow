#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_XLA_NODE_ATTR_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_XLA_NODE_ATTR_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/engine/xla/of2xla/xla_node.h"

namespace oneflow {
namespace mla {

template <typename T>
T GetNodeAttr(const XlaNode *node, const std::string &attr_name) {
  DCHECK(HasFieldInPbMessage(node->proto_conf(), attr_name));
  return GetValFromPbMessage<T>(node->proto_conf(), attr_name);
}

template <>
Shape GetNodeAttr<Shape>(const XlaNode *node, const std::string &attr_name);

typedef oneflow::PbMessage *PbMessagePtr;
template <>
PbMessagePtr GetNodeAttr<PbMessagePtr>(const XlaNode *node,
                                       const std::string &attr_name);

std::string GetNodeAttrAsString(const XlaNode *node,
                                const std::string &attr_name);

}  // namespace mla
}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_XLA_OF2XLA_XLA_NODE_ATTR_H_
