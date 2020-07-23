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
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

class TransposeOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    const auto& perm = ctx->Attr<std::vector<int32_t>>("perm");
    Shape x_shape = ctx->SoleInputShape();
    CHECK_EQ(perm.size(), x_shape.NumAxes());

    xla::XlaOp x = ctx->SoleInput();
    if (IsIdentity(perm)) {
      ctx->SetSoleOutput(x);
    } else {
      std::vector<long long> transposed_order(x_shape.NumAxes());
      for (int i = 0; i < x_shape.NumAxes(); ++i) { transposed_order[i] = perm[i]; }
      ctx->SetSoleOutput(xla::Transpose(x, transposed_order));
    }
  }

  bool IsIdentity(const std::vector<int32_t> &perm) const {
    bool is_identity = true;
    for (int i = 0; i < perm.size(); ++i) {
      if (i != perm[i]) {
        is_identity = false;
        break;
      }
    }
    return is_identity || (perm.size() <= 1);
  }
};

REGISTER_XLA_OP_KERNEL(Transpose, TransposeOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
