#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

nvinfer1::MatrixOperation GetMatrixOperation(nvinfer1::ITensor *x, bool transpose) {
  if (x->getDimensions().nbDims < 2) { return nvinfer1::MatrixOperation::kVECTOR; }
  return transpose ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
}

class MatMulOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape a_shape = ctx->InputShape("a_0");
    Shape b_shape = ctx->InputShape("b_0");
    CHECK_GE(a_shape.NumAxes(), 2);
    CHECK_EQ(a_shape.NumAxes(), b_shape.NumAxes());

    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");
    nvinfer1::ITensor *a = ctx->Input("a_0");
    nvinfer1::ITensor *b = ctx->Input("b_0");

    auto op0 = GetMatrixOperation(a, transpose_a);
    auto op1 = GetMatrixOperation(b, transpose_b);

    auto *layer = ctx->builder()->addMatrixMultiply(*a, op0, *b, op1);
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(MatMul, MatMulOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
