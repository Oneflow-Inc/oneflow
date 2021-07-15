/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/xrt/node_util.h"
#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/message_attr.h"
#include "oneflow/xrt/utility/registry.h"

namespace oneflow {
namespace xrt {

const PbMessage* OpMessage(const XrtNode* node) {
  const PbMessage* message = &node->param();
  if (!node->IsArgumentNode()) {
    util::GetOneofMessage(node->param(), "op_type", &message);
    CHECK(message) << "Cann't get op_type message in node which name is " << node->name();
  }
  return message;
}

bool IsCompiledNode(const XrtNode* node, const XrtEngine& engine, const bool train_phase) {
  auto field = MakeXrtField(node->device(), engine);
  return OpKernelRegistered(node->type(), field)
         && (!train_phase || TrainPhaseEnabled(node->type(), field));
}

bool IsOptimizerNode(const XrtNode* node, const XrtEngine& engine) {
  auto field = MakeXrtField(node->device(), engine);
  return IsOptimizerOp(node->type(), field);
}

bool IsNodeInput(const XrtNode* node, const Argument& argument) {
  for (XrtEdge* edge : node->in_edges()) {
    if (edge->argument() == argument) { return true; }
  }
  return false;
}

bool IsNodeOutput(const XrtNode* node, const Argument& argument) {
  for (XrtEdge* edge : node->out_edges()) {
    if (edge->argument() == argument) { return true; }
  }
  return false;
}

}  // namespace xrt
}  // namespace oneflow
