#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<typename T>
static T *GetWeightPtr(const nvinfer1::Weights &weight) {
  return reinterpret_cast<T *>(const_cast<void *>(weight.values));
}

class NormalizationOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->InputShape("in");
    CHECK_GE(in_shape.NumAxes(), 2);

    float epsilon = ctx->Attr<float>("epsilon");

    nvinfer1::Weights gamma = ctx->Weight("gamma");
    nvinfer1::Weights beta = ctx->Weight("beta");
    nvinfer1::Weights moving_mean = ctx->Weight("moving_mean");
    nvinfer1::Weights moving_variance = ctx->Weight("moving_variance");

    float *gamma_ptr = GetWeightPtr<float>(gamma);
    float *beta_ptr = GetWeightPtr<float>(beta);
    const float *moving_mean_ptr = GetWeightPtr<float>(moving_mean);
    const float *moving_variance_ptr = GetWeightPtr<float>(moving_variance);

    for (int i = 0; i < gamma.count; ++i) {
      *gamma_ptr /= std::sqrt(*moving_variance_ptr + epsilon);
      *beta_ptr -= *moving_mean_ptr * (*gamma_ptr);
      gamma_ptr += 1;
      beta_ptr += 1;
      moving_mean_ptr += 1;
      moving_variance_ptr += 1;
    }

    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::ITensor *input = ctx->Input("in");

    nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
    nvinfer1::IScaleLayer *layer =  // NOLINT
        ctx->builder()->addScale(*input, mode, beta, gamma, power);
    layer->setName(ctx->op_name().c_str());

    ctx->SetOutput("out", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Normalization, NormalizationOp).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
