#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

#define DEFINE_CUDNN_DATA_TYPE(type_cpp, type_cudnn)                          \
  const cudnnDataType_t CudnnDataType<type_cpp>::val = type_cudnn;            \
  const type_cpp CudnnDataType<type_cpp>::oneval = static_cast<type_cpp>(1);  \
  const type_cpp CudnnDataType<type_cpp>::zeroval = static_cast<type_cpp>(0); \
  const void* CudnnDataType<type_cpp>::one = &oneval;                         \
  const void* CudnnDataType<type_cpp>::zero = &zeroval;
OF_PP_FOR_EACH_TUPLE(DEFINE_CUDNN_DATA_TYPE, CUDNN_DATA_TYPE_SEQ);

cudnnDataType_t GetCudnnDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_cudnn) \
  if (val == GetDataType<type_cpp>::val) { return type_cudnn; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CUDNN_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNEXPECTED_RUN();
}

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

CudnnTensorNdDesc::~CudnnTensorNdDesc() {
  CudaCheck(cudnnDestroyTensorDescriptor(val_));
}

CudnnTensorNdDesc::CudnnTensorNdDesc(DataType data_type,
                                     const std::vector<int>& dim,
                                     const std::vector<int>& stride) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(data_type),
                                       dim.size(), dim.data(), stride.data()));
}

CudnnPoolingNdDesc::~CudnnPoolingNdDesc() {
  CudaCheck(cudnnDestroyPoolingDescriptor(val_));
}

CudnnPoolingNdDesc::CudnnPoolingNdDesc(PoolingMode pooling_mode,
                                       const std::vector<int>& window,
                                       const std::vector<int>& padding,
                                       const std::vector<int>& stride) {
  CudaCheck(cudnnCreatePoolingDescriptor(&val_));
  CudaCheck(cudnnSetPoolingNdDescriptor(
      val_,
      (pooling_mode == PoolingMode::kAveragePooling
           ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
           : CUDNN_POOLING_MAX),
      CUDNN_NOT_PROPAGATE_NAN, window.size(), window.data(), padding.data(),
      stride.data()));
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

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, const Shape& shape)
    : CudnnFilterDesc(data_type, shape.At(0), shape.At(1), shape.At(2),
                      shape.At(3)) {
  CHECK_EQ(shape.NumAxes(), 4);
}

CudnnActivationDesc::CudnnActivationDesc(cudnnActivationMode_t mode,
                                         cudnnNanPropagation_t relu_nan_opt,
                                         double coef) {
  CudaCheck(cudnnCreateActivationDescriptor(&val_));
  CudaCheck(cudnnSetActivationDescriptor(val_, mode, relu_nan_opt, coef));
}

CudnnActivationDesc::~CudnnActivationDesc() {
  CudaCheck(cudnnDestroyActivationDescriptor(val_));
}

}  // namespace oneflow
