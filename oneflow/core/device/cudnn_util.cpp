#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

#ifdef WITH_CUDA

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
  UNIMPLEMENTED();
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

CudnnTensorDesc::CudnnTensorDesc(DataType data_type,
                                 const std::vector<int>& dim,
                                 const std::vector<int>& stride) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(data_type),
                                       dim.size(), dim.data(), stride.data()));
}

CudnnFilterDesc::~CudnnFilterDesc() {
  CudaCheck(cudnnDestroyFilterDescriptor(val_));
}

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, const Shape& shape,
                                 const std::string& data_format) {
  cudnnTensorFormat_t cudnn_data_format;
  if (data_format == "channel_first") {
    cudnn_data_format = CUDNN_TENSOR_NCHW;
  } else if (data_format == "channel_last") {
    cudnn_data_format = CUDNN_TENSOR_NHWC;
  } else {
    UNIMPLEMENTED();
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

#endif  // WITH_CUDA

}  // namespace oneflow
