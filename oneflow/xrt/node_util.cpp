#include "oneflow/xrt/node_util.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/message_attr.h"
#include "oneflow/xrt/utility/registry.h"

namespace oneflow {
namespace xrt {

const PbMessage *OpMessage(const XrtNode *node) {
  const PbMessage *message = &node->param();
  if (!node->IsArgumentNode()) {
    util::GetOneofMessage(node->param(), "op_type", &message);
    CHECK(message) << "Cann't get op_type message in node which name is "
                   << node->name();
  }
  return message;
}

bool IsNodeCompiled(const XrtNode *node, const XrtEngine &engine) {
  auto field = MakeXrtField(node->device(), engine);
  auto *rm = util::RegistryManager<XrtField>::Global();
  if (rm->HasField(field)) {
    return rm->Get(field)->IsRegistered(node->type());
  }
  return false;
}

bool IsNodeInput(const XrtNode *node, const Argument &argument) {
  for (XrtEdge *edge : node->in_edges()) {
    if (edge->argument() == argument) {
      return true;
    }
  }
  return false;
}

bool IsNodeOutput(const XrtNode *node, const Argument &argument) {
  for (XrtEdge *edge : node->out_edges()) {
    if (edge->argument() == argument) {
      return true;
    }
  }
  return false;
}

}  // namespace xrt
}  // namespace oneflow
