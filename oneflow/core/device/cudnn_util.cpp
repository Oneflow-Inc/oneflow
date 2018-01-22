#ifdef WITH_CUDNN

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

float CudnnDataType<float>::oneval = 1.0f;
float CudnnDataType<float>::zeroval = 0.0f;
const void* CudnnDataType<float>::one =
    static_cast<void*>(&CudnnDataType<float>::oneval);
const void* CudnnDataType<float>::zero =
    static_cast<void*>(&CudnnDataType<float>::zeroval);

double CudnnDataType<double>::oneval = 1.0;
double CudnnDataType<double>::zeroval = 0.0;
const void* CudnnDataType<double>::one =
    static_cast<void*>(&CudnnDataType<double>::oneval);
const void* CudnnDataType<double>::zero =
    static_cast<void*>(&CudnnDataType<double>::zeroval);

CudnnTensorDesc::~CudnnTensorDesc() {
  CudaCheck(cudnnDestroyTensorDescriptor(val_));
}

CudnnTensorDesc::CudnnTensorDesc(DataType data_type, int64_t n, int64_t c,
                                 int64_t h, int64_t w) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensor4dDescriptor(
      val_, CUDNN_TENSOR_NCHW, GetCudnnDataType(data_type), n, c, h, w));
}

CudnnTensorDesc::CudnnTensorDesc(DataType data_type, const Shape& shape) {
  CHECK_EQ(shape.NumAxes(), 4);
  CudnnTensorDesc(data_type, shape.At(0), shape.At(1), shape.At(2),
                  shape.At(3));
}

CudnnFilterDesc::~CudnnFilterDesc() {
  CudaCheck(cudnnDestroyFilterDescriptor(val_));
}

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, int64_t k, int64_t c,
                                 int64_t h, int64_t w) {
  CudaCheck(cudnnCreateFilterDescriptor(&val_));
  CudaCheck(cudnnSetFilter4dDescriptor(val_, GetCudnnDataType(data_type),
                                       CUDNN_TENSOR_NCHW, k, c, h, w));
}

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, const Shape& shape) {
  CHECK_EQ(shape.NumAxes(), 4);
  CudnnFilterDesc(data_type, shape.At(0), shape.At(1), shape.At(2),
                  shape.At(3));
}

CudnnConvolutionDesc::~CudnnConvolutionDesc() {
  CudaCheck(cudnnDestroyConvolutionDescriptor(val_));
}

CudnnConvolutionDesc::CudnnConvolutionDesc(DataType data_type, int64_t pad_h,
                                           int64_t pad_w, int64_t stride_h,
                                           int64_t stride_w,
                                           int64_t dilation_h = 1,
                                           int64_t dilation_w = 1) {
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  CudaCheck(cudnnSetConvolution2dDescriptor(
      val_, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      CUDNN_CROSS_CORRELATION, GetCudnnDataType(data_type)));
}

CudnnConvolutionDesc::CudnnConvolutionDesc(DataType data_type,
                                           const ConvolutionOpConf& conv_conf)
    : CudnnConvolutionDesc(data_type, conv_conf.pad_h(), conv_conf.pad_w(),
                           conv_conf.stride_h(), conv_conf.stride_w()) {}

}  // namespace oneflow

#endif  // WITH_CUDNN
