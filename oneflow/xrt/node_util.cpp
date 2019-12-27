#include "oneflow/xrt/node_util.h"
#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/message_attr.h"
#include "oneflow/xrt/utility/registry.h"

namespace oneflow {
namespace xrt {

const PbMessage *OpMessage(const XrtNode *node) {
  const PbMessage *message = &node->param();
  if (!node->IsArgumentNode()) {
    util::GetOneofMessage(node->param(), "op_type", &message);
    CHECK(message) << "Cann't get op_type message in node which name is " << node->name();
  }
  return message;
}

bool IsCompiledNode(const XrtNode *node, const XrtEngine &engine, const bool train_phase) {
  auto field = MakeXrtField(node->device(), engine);
  return OpKernelRegistered(node->type(), field)
         && (!train_phase || TrainPhaseEnabled(node->type(), field));
}

bool IsOptimizerNode(const XrtNode *node, const XrtEngine &engine) {
  auto field = MakeXrtField(node->device(), engine);
  return IsOptimizerOp(node->type(), field);
}

bool IsNodeInput(const XrtNode *node, const Argument &argument) {
  for (XrtEdge *edge : node->in_edges()) {
    if (edge->argument() == argument) { return true; }
  }
  return false;
}

bool IsNodeOutput(const XrtNode *node, const Argument &argument) {
  for (XrtEdge *edge : node->out_edges()) {
    if (edge->argument() == argument) { return true; }
  }
  return false;
}

}  // namespace xrt
}  // namespace oneflow
