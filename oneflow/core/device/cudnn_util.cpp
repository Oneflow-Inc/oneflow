#ifdef WITH_CUDNN

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

namespace {

cudnnDataType_t GetCudnnDataType(DataType val) {
  switch (val) {
    case DataType::kFloat: return CUDNN_DATA_FLOAT;
    case DataType::kDouble: return CUDNN_DATA_DOUBLE;
    case DataType::kInt8: return CUDNN_DATA_INT8;
    case DataType::kInt32: return CUDNN_DATA_INT32;
    default: UNEXPECTED_RUN();
  }
}

}  // namespace

CudnnTensorDesc::~CudnnTensorDesc() {
  CudaCheck(cudnnDestroyTensorDescriptor(val_));
}
CudnnTensorDesc::CudnnTensorDesc(DataType data_type, int n, int c, int h,
                                 int w) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensor4dDescriptor(
      val_, CUDNN_TENSOR_NCHW, GetCudnnDataType(data_type), n, c, h, w));
}
CudnnTensorDesc::CudnnTensorDesc(DataType data_type, const Shape& shape)
    : CudnnTensorDesc(data_type, shape.At(0), shape.At(1), shape.At(2),
                      shape.At(3)) {
  CHECK_EQ(shape.NumAxes(), 4);
}

CudnnFilterDesc::~CudnnFilterDesc() {
  CudaCheck(cudnnDestroyFilterDescriptor(val_));
}
CudnnFilterDesc::CudnnFilterDesc(DataType data_type, int k, int c, int h,
                                 int w) {
  CudaCheck(cudnnCreateFilterDescriptor(&val_));
  cudnnSetFilter4dDescriptor(val_, GetCudnnDataType(data_type),
                             CUDNN_TENSOR_NCHW, k, c, h, w);
}

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, const Shape& shape)
    : CudnnFilterDesc(data_type, shape.At(0), shape.At(1), shape.At(2),
                      shape.At(3)) {
  CHECK_EQ(shape.NumAxes(), 4);
}

}  // namespace oneflow

#endif  // WITH_CUDNN
