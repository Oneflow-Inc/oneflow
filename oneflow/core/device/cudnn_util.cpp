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

CudnnTensorNdDesc::~CudnnTensorNdDesc() {
  CudaCheck(cudnnDestroyTensorDescriptor(val_));
}

CudnnTensorNdDesc::CudnnTensorNdDesc(DataType data_type, const Shape& shape) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  std::vector<int> stride_of_tensor(shape.NumAxes(), 1);
  CudaCheck(cudnnSetTensorNdDescriptor(
      val_, GetCudnnDataType(data_type), static_cast<int>(shape.NumAxes()),
      reinterpret_cast<const int*>(shape.dim_vec().data()),
      stride_of_tensor.data()));
}

CudnnFilterNdDesc::~CudnnFilterNdDesc() {
  CudaCheck(cudnnDestroyFilterDescriptor(val_));
}

CudnnFilterNdDesc::CudnnFilterNdDesc(DataType data_type,
                                     std::string data_format,
                                     const Shape& shape) {
  std::vector<int> stride_of_tensor(shape.NumAxes(), 1);
  cudnnTensorFormat_t cudnn_data_format;
  if (data_format == "NCHW" || data_format == "NCDHW") {
    cudnn_data_format = CUDNN_TENSOR_NCHW;
  } else if (data_format == "NHWC" || data_format == "NDHWC") {
    cudnn_data_format = CUDNN_TENSOR_NHWC;
  } else {
    UNEXPECTED_RUN();
  }
  CudaCheck(cudnnSetFilterNdDescriptor(
      val_, GetCudnnDataType(data_type), cudnn_data_format,
      static_cast<int>(shape.NumAxes()),
      reinterpret_cast<const int*>(shape.dim_vec().data())));
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
