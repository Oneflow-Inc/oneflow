#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

nvinfer1::MatrixOperation GetMatrixOperation(nvinfer1::ITensor *x,
                                             bool transpose) {
  if (x->getDimensions().nbDims < 2) {
    return nvinfer1::MatrixOperation::kVECTOR;
  }
  return transpose ? nvinfer1::MatrixOperation::kTRANSPOSE
                   : nvinfer1::MatrixOperation::kNONE;
}

class SoftMaxOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");
    CHECK_GE(in_shape.NumAxes(), 2);

    int32_t axis = ctx->GetAttr<int32_t>("axis");
    CHECK_GE(axis, 1);
    CHECK_LT(axis, in_shape.NumAxes() - 1);
    //dims = in_shape.NumAxes()
    Shape transposed_shape = in_shape;
    nvinfer1::ITensor *in = ctx->Input("in");
    bool need_transpose =false; 
    axis == in_shape.NumAxes() - 1 ? need_transpose = false : true;
    if(need_transpose) {
      transposed_shape = in_shape.Set(axis, in_shape.At(in_shape.NumAxes() - 1));
      transposed_shape = transposed_shape.Set(in_shape.NumAxes() - 1, in_shape.At(axis)); 
    }
    auto *layer = ctx->builder()->addSoftMax(*GetMatrixOperation(*in, need_transpose));
    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(SoftMax, SoftMaxOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
