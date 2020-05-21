#ifndef ONEFLOW_XRT_NODE_UTIL_H_
#define ONEFLOW_XRT_NODE_UTIL_H_

#include "oneflow/core/common/protobuf.h"

#include "oneflow/xrt/graph/node.h"

namespace oneflow {
namespace xrt {

const PbMessage *OpMessage(const XrtNode *node);

bool IsCompiledNode(const XrtNode *node, const XrtEngine &engine, const bool train_phase);
bool IsOptimizerNode(const XrtNode *node, const XrtEngine &engine);

bool IsNodeInput(const XrtNode *node, const Argument &argument);
bool IsNodeOutput(const XrtNode *node, const Argument &argument);

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_NODE_UTIL_H_
