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
#include "oneflow/core/operator/copy_comm_net_op.h"

namespace oneflow {

void CopyCommNetOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

LogicalBlobId CopyCommNetOp::lbi4obn(const std::string& output_bn) const { return GenPackedLbi(); }

LogicalBlobId CopyCommNetOp::lbi4ibn(const std::string& input_bn) const { return GenPackedLbi(); }

REGISTER_OP(OperatorConf::kCopyCommNetConf, CopyCommNetOp);

}  // namespace oneflow
