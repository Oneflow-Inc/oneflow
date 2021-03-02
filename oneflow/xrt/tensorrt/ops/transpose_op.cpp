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
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class TransposeOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    const auto &perm = ctx->Attr<std::vector<int32_t>>("perm");
    Shape in_shape = ctx->SoleInputShape();
    CHECK_EQ(perm.size(), in_shape.NumAxes());

    nvinfer1::ITensor *input = ctx->SoleInput();
    if (IsIdentity(perm)) {
      ctx->SetSoleOutput(input);
    } else {
      ctx->SetSoleOutput(helpers::Transpose(ctx, input, perm));
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

REGISTER_TRT_OP_KERNEL(Transpose, TransposeOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
