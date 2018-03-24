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
CudnnTensorDesc::CudnnTensorDesc(cudnnTensorFormat_t format, DataType data_type,
                                 int n, int c, int h, int w) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensor4dDescriptor(
      val_, format, GetCudnnDataType(data_type), n, c, h, w));
}
CudnnTensorDesc::CudnnTensorDesc(const std::string& data_format, DataType type,
                                 int size, int n, int c, int h, int w, int d) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  if (data_format == "channels_last") {
    if (size == 4) {
      CudaCheck(cudnnSetTensor4dDescriptorEx(val_, GetCudnnDataType(type), n, c,
                                             h, w, h * w * c, 1, w * c, c));
    } else {
      // need check
      std::vector<int> dims = {n, h, w, d, c};
      std::vector<int> strides = {h * w * d * c, w * d * c, d * c, c, 1};
      CudaCheck(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(type),
                                           size > 3 ? size : 4, dims.data(),
                                           strides.data()));
    }
  } else if (data_format == "channels_first") {
    if (size == 4) {
      CudaCheck(cudnnSetTensor4dDescriptorEx(val_, GetCudnnDataType(type), n, c,
                                             h, w, c * h * w, h * w, w, 1));
    } else {
      std::vector<int> dims = {n, c, h, w, d};
      std::vector<int> strides = {c * h * w * d, h * w * d, w * d, d, 1};
      CudaCheck(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(type),
                                           size > 3 ? size : 4, dims.data(),
                                           strides.data()));
    }
  } else {
    LOG(FATAL) << "Unknown data format";
  }
}
CudnnTensorDesc::CudnnTensorDesc(DataType data_type, int dims, const int* dim,
                                 const int* stride) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(data_type), dims,
                                       dim, stride));
}

CudnnFilterDesc::~CudnnFilterDesc() {
  CudaCheck(cudnnDestroyFilterDescriptor(val_));
}

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, const Shape& shape,
                                 const std::string& data_format) {
  CudaCheck(cudnnCreateFilterDescriptor(&val_));
  cudnnTensorFormat_t cudnn_data_format;
  if (data_format == "channels_first") {
    cudnn_data_format = CUDNN_TENSOR_NCHW;
  } else if (data_format == "channels_last") {
    cudnn_data_format = CUDNN_TENSOR_NHWC;
  } else {
    UNIMPLEMENTED();
  }

  if (shape.NumAxes() == 4) {
    int filters = static_cast<int>(shape.At(0));
    int kernel_h = data_format == "channels_first"
                       ? static_cast<int>(shape.At(2))
                       : static_cast<int>(shape.At(1));
    int kernel_w = data_format == "channels_first"
                       ? static_cast<int>(shape.At(3))
                       : static_cast<int>(shape.At(2));
    int c = data_format == "channels_first" ? static_cast<int>(shape.At(1))
                                            : static_cast<int>(shape.At(3));
    CudaCheck(cudnnSetFilter4dDescriptor(val_, GetCudnnDataType(data_type),
                                         cudnn_data_format, filters, c,
                                         kernel_h, kernel_w));
  } else {
    std::vector<int> dims(shape.dim_vec().begin(), shape.dim_vec().end());
    CudaCheck(cudnnSetFilterNdDescriptor(val_, GetCudnnDataType(data_type),
                                         cudnn_data_format, dims.size(),
                                         dims.data()));
  }
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
