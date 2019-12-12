#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

#ifdef WITH_CUDA

cudnnDataType_t GetCudnnDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_cudnn) \
  if (val == GetDataType<type_cpp>::value) { return type_cudnn; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CUDNN_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNIMPLEMENTED();
}

CudnnTensorDesc::CudnnTensorDesc() { CudaCheck(cudnnCreateTensorDescriptor(&val_)); }
CudnnTensorDesc::~CudnnTensorDesc() { CudaCheck(cudnnDestroyTensorDescriptor(val_)); }
CudnnTensorDesc::CudnnTensorDesc(cudnnTensorFormat_t format, DataType data_type, int n, int c,
                                 int h, int w) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensor4dDescriptor(val_, format, GetCudnnDataType(data_type), n, c, h, w));
}
CudnnTensorDesc::CudnnTensorDesc(DataType data_type, int dims, const int* dim, const int* stride) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(data_type), dims, dim, stride));
}
CudnnTensorDesc::CudnnTensorDesc(DataType data_type, const DenseShapeView& shape,
                                 const std::string& data_format) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  cudnnTensorFormat_t cudnn_data_format;
  if (data_format == "channels_first") {
    cudnn_data_format = CUDNN_TENSOR_NCHW;
  } else if (data_format == "channels_last") {
    cudnn_data_format = CUDNN_TENSOR_NHWC;
  } else {
    UNIMPLEMENTED();
  }

  if (shape.NumAxes() == 3) {
    int data_num = static_cast<int>(shape.At(0));
    int channels = data_format == "channels_first" ? static_cast<int>(shape.At(1))
                                                   : static_cast<int>(shape.At(2));
    int kernel_h = data_format == "channels_first" ? static_cast<int>(shape.At(2))
                                                   : static_cast<int>(shape.At(1));
    int kernel_w = 1;
    CudaCheck(cudnnSetTensor4dDescriptor(val_, cudnn_data_format, GetCudnnDataType(data_type),
                                         data_num, channels, kernel_h, kernel_w));
  } else if (shape.NumAxes() == 4) {
    int data_num = static_cast<int>(shape.At(0));
    int channels = data_format == "channels_first" ? static_cast<int>(shape.At(1))
                                                   : static_cast<int>(shape.At(3));
    int kernel_h = data_format == "channels_first" ? static_cast<int>(shape.At(2))
                                                   : static_cast<int>(shape.At(1));
    int kernel_w = data_format == "channels_first" ? static_cast<int>(shape.At(3))
                                                   : static_cast<int>(shape.At(2));
    CudaCheck(cudnnSetTensor4dDescriptor(val_, cudnn_data_format, GetCudnnDataType(data_type),
                                         data_num, channels, kernel_h, kernel_w));
  } else {
    std::vector<int> tensor_dim({shape.ptr(), shape.ptr() + shape.NumAxes()});
    std::vector<int> stride_of_tensor(shape.NumAxes(), 1);
    for (int32_t i = shape.NumAxes() - 2; i >= 0; --i) {
      stride_of_tensor[i] = stride_of_tensor[i + 1] * shape.At(i + 1);
    }

    CudaCheck(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(data_type), shape.NumAxes(),
                                         tensor_dim.data(), stride_of_tensor.data()));
  }
}

CudnnFilterDesc::~CudnnFilterDesc() { CudaCheck(cudnnDestroyFilterDescriptor(val_)); }

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, const DenseShapeView& shape,
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

  if (shape.NumAxes() == 3) {
    int filters = static_cast<int>(shape.At(0));
    int c = data_format == "channels_first" ? static_cast<int>(shape.At(1))
                                            : static_cast<int>(shape.At(2));
    int kernel_h = data_format == "channels_first" ? static_cast<int>(shape.At(2))
                                                   : static_cast<int>(shape.At(1));
    int kernel_w = 1;
    CudaCheck(cudnnSetFilter4dDescriptor(val_, GetCudnnDataType(data_type), cudnn_data_format,
                                         filters, c, kernel_h, kernel_w));
  } else if (shape.NumAxes() == 4) {
    int filters = static_cast<int>(shape.At(0));
    int kernel_h = data_format == "channels_first" ? static_cast<int>(shape.At(2))
                                                   : static_cast<int>(shape.At(1));
    int kernel_w = data_format == "channels_first" ? static_cast<int>(shape.At(3))
                                                   : static_cast<int>(shape.At(2));
    int c = data_format == "channels_first" ? static_cast<int>(shape.At(1))
                                            : static_cast<int>(shape.At(3));
    CudaCheck(cudnnSetFilter4dDescriptor(val_, GetCudnnDataType(data_type), cudnn_data_format,
                                         filters, c, kernel_h, kernel_w));
  } else {
    std::vector<int> dims({shape.ptr(), shape.ptr() + shape.NumAxes()});
    CudaCheck(cudnnSetFilterNdDescriptor(val_, GetCudnnDataType(data_type), cudnn_data_format,
                                         dims.size(), dims.data()));
  }
}

CudnnActivationDesc::CudnnActivationDesc(cudnnActivationMode_t mode,
                                         cudnnNanPropagation_t relu_nan_opt, double coef) {
  CudaCheck(cudnnCreateActivationDescriptor(&val_));
  CudaCheck(cudnnSetActivationDescriptor(val_, mode, relu_nan_opt, coef));
}

CudnnActivationDesc::~CudnnActivationDesc() { CudaCheck(cudnnDestroyActivationDescriptor(val_)); }

#endif  // WITH_CUDA

template<typename T>
const void* CudnnSPOnePtr() {
  static const float fval = 1.0f;
  static const double dval = 1.0;
  const void* ret = std::is_same<T, double>::value ? static_cast<const void*>(&dval)
                                                   : static_cast<const void*>(&fval);
  return ret;
}

template<typename T>
const void* CudnnSPZeroPtr() {
  static const float fval = 0.0f;
  static const double dval = 0.0;
  const void* ret = std::is_same<T, double>::value ? static_cast<const void*>(&dval)
                                                   : static_cast<const void*>(&fval);
  return ret;
}

template const void* CudnnSPOnePtr<float>();
template const void* CudnnSPOnePtr<double>();
template const void* CudnnSPOnePtr<float16>();

template const void* CudnnSPZeroPtr<float>();
template const void* CudnnSPZeroPtr<double>();
template const void* CudnnSPZeroPtr<float16>();

}  // namespace oneflow
