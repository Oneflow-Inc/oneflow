#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class TransposeOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    auto perm = ctx->GetAttr<std::vector<int32_t>>("perm");
    Shape in_shape = ctx->InputShape("in");
    CHECK_EQ(perm.size(), in_shape.NumAxes());

    nvinfer1::ITensor *input = ctx->Input("in");
    if (IsIdentity(perm)) {
      ctx->SetOutput("out", input);
    } else {
      ctx->SetOutput("out", helpers::Transpose(ctx, input, perm));
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
