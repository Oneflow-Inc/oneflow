#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

namespace helpers {

bool DimsEqual(const nvinfer1::Dims &dim1, const nvinfer1::Dims &dim2) {
  if (dim1.nbDims != dim2.nbDims) {
    return false;
  }
  for (int i = 0; i < dim1.nbDims; ++i) {
    if (dim1.d[i] != dim2.d[i]) {
      return false;
    }
  }
  return true;
}

nvinfer1::ITensor *Reshape(TrtOpContext *ctx, nvinfer1::ITensor *in,  // NOLINT
                           const Shape &shape) {
  nvinfer1::Dims dims = ShapeToXrtDims(shape);
  if (DimsEqual(in->getDimensions(), dims)) {
    return in;
  }
  auto *layer = ctx->builder()->addShuffle(*in);
  layer->setReshapeDimensions(dims);
  return layer->getOutput(0);
}

nvinfer1::ITensor *Reshape(TrtOpContext *ctx, nvinfer1::Weights in,  // NOLINT
                           const Shape &shape) {
  nvinfer1::Dims dims = ShapeToXrtDims(shape);
  auto *layer = ctx->builder()->addConstant(dims, in);
  return layer->getOutput(0);
}

}  // namespace helpers

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
