#ifdef WITH_CUDNN

#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

cudnnDataType_t GetCudnnDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_cudnn) \
  if (val == GetDataType<type_cpp>::val) { return type_cudnn; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CUDNN_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNEXPECTED_RUN();
}  // namespace oneflow

CudnnTensorDesc::~CudnnTensorDesc() {
  CudaCheck(cudnnDestroyTensorDescriptor(val_));
}

CudnnTensorDesc::CudnnTensorDesc(DataType data_type, int n, int c, int h,
                                 int w) {
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

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, int k, int c, int h,
                                 int w) {
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

CudnnConvolutionDesc::CudnnConvolutionDesc(DataType data_type,
                                           const Conv2dOpConf& conv2d_conf) {
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  CudaCheck(cudnnSetConvolution2dDescriptor(
      val_, conv2d_conf.pad_h(), conv2d_conf.pad_w(), conv2d_conf.stride_h(),
      conv2d_conf.stride_w(), conv2d_conf.dilation_h(),
      conv2d_conf.dilation_w(), CUDNN_CROSS_CORRELATION,
      GetCudnnDataType(data_type)));
}

}  // namespace oneflow

#endif  // WITH_CUDNN
