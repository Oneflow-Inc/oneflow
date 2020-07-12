#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ReshapeOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    Shape shape = ctx->SoleOutputShape();
    CHECK_EQ(shape.Count(0), in_shape.Count(0));

    nvinfer1::ITensor *input = ctx->SoleInput();
    ctx->SetSoleOutput(helpers::Reshape(ctx, input, shape));
  }
};

REGISTER_TRT_OP_KERNEL(Reshape, ReshapeOp).EnableTrainPhase().Finalize();

class ReshapeLikeOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape x_shape = ctx->InputShape("in_0");
    Shape like_shape = ctx->InputShape("like_0");
    CHECK_EQ(x_shape.Count(0), like_shape.Count(0));

    nvinfer1::ITensor *input = ctx->Input("in_0");
    ctx->SetSoleOutput(helpers::Reshape(ctx, input, like_shape));
  }
};

REGISTER_TRT_OP_KERNEL(ReshapeLike, ReshapeLikeOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
