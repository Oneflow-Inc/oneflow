/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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

CudnnTensorDesc::CudnnTensorDesc() { OF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&val_)); }
CudnnTensorDesc::~CudnnTensorDesc() { OF_CUDNN_CHECK(cudnnDestroyTensorDescriptor(val_)); }
CudnnTensorDesc::CudnnTensorDesc(cudnnTensorFormat_t format, DataType data_type, int n, int c,
                                 int h, int w) {
  OF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&val_));
  OF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(val_, format, GetCudnnDataType(data_type), n, c, h, w));
}
CudnnTensorDesc::CudnnTensorDesc(DataType data_type, int dims, const int* dim, const int* stride) {
  OF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&val_));
  OF_CUDNN_CHECK(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(data_type), dims, dim, stride));
}
CudnnTensorDesc::CudnnTensorDesc(DataType data_type, const ShapeView& shape,
                                 const std::string& data_format) {
  OF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&val_));
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
    OF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(val_, cudnn_data_format, GetCudnnDataType(data_type),
                                              data_num, channels, kernel_h, kernel_w));
  } else if (shape.NumAxes() == 4) {
    int data_num = static_cast<int>(shape.At(0));
    int channels = data_format == "channels_first" ? static_cast<int>(shape.At(1))
                                                   : static_cast<int>(shape.At(3));
    int kernel_h = data_format == "channels_first" ? static_cast<int>(shape.At(2))
                                                   : static_cast<int>(shape.At(1));
    int kernel_w = data_format == "channels_first" ? static_cast<int>(shape.At(3))
                                                   : static_cast<int>(shape.At(2));
    OF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(val_, cudnn_data_format, GetCudnnDataType(data_type),
                                              data_num, channels, kernel_h, kernel_w));
  } else {
    std::vector<int> tensor_dim({shape.ptr(), shape.ptr() + shape.NumAxes()});
    std::vector<int> stride_of_tensor(shape.NumAxes(), 1);
    for (int32_t i = shape.NumAxes() - 2; i >= 0; --i) {
      stride_of_tensor[i] = stride_of_tensor[i + 1] * shape.At(i + 1);
    }

    OF_CUDNN_CHECK(cudnnSetTensorNdDescriptor(val_, GetCudnnDataType(data_type), shape.NumAxes(),
                                              tensor_dim.data(), stride_of_tensor.data()));
  }
}

CudnnFilterDesc::~CudnnFilterDesc() { OF_CUDNN_CHECK(cudnnDestroyFilterDescriptor(val_)); }

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, const ShapeView& shape,
                                 const std::string& data_format) {
  OF_CUDNN_CHECK(cudnnCreateFilterDescriptor(&val_));
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
    OF_CUDNN_CHECK(cudnnSetFilter4dDescriptor(val_, GetCudnnDataType(data_type), cudnn_data_format,
                                              filters, c, kernel_h, kernel_w));
  } else if (shape.NumAxes() == 4) {
    int filters = static_cast<int>(shape.At(0));
    int kernel_h = data_format == "channels_first" ? static_cast<int>(shape.At(2))
                                                   : static_cast<int>(shape.At(1));
    int kernel_w = data_format == "channels_first" ? static_cast<int>(shape.At(3))
                                                   : static_cast<int>(shape.At(2));
    int c = data_format == "channels_first" ? static_cast<int>(shape.At(1))
                                            : static_cast<int>(shape.At(3));
    OF_CUDNN_CHECK(cudnnSetFilter4dDescriptor(val_, GetCudnnDataType(data_type), cudnn_data_format,
                                              filters, c, kernel_h, kernel_w));
  } else {
    std::vector<int> dims({shape.ptr(), shape.ptr() + shape.NumAxes()});
    OF_CUDNN_CHECK(cudnnSetFilterNdDescriptor(val_, GetCudnnDataType(data_type), cudnn_data_format,
                                              dims.size(), dims.data()));
  }
}

CudnnActivationDesc::CudnnActivationDesc(cudnnActivationMode_t mode,
                                         cudnnNanPropagation_t relu_nan_opt, double coef) {
  OF_CUDNN_CHECK(cudnnCreateActivationDescriptor(&val_));
  OF_CUDNN_CHECK(cudnnSetActivationDescriptor(val_, mode, relu_nan_opt, coef));
}

CudnnActivationDesc::~CudnnActivationDesc() {
  OF_CUDNN_CHECK(cudnnDestroyActivationDescriptor(val_));
}

size_t GetCudnnDataTypeByteSize(cudnnDataType_t data_type) {
  size_t byte_size = 0;
  switch (data_type) {
    case CUDNN_DATA_FLOAT:
    case CUDNN_DATA_INT32:
    case CUDNN_DATA_INT8x4:
    case CUDNN_DATA_UINT8x4: {
      byte_size = 4;
      break;
    }
    case CUDNN_DATA_DOUBLE: {
      byte_size = 8;
      break;
    }
    case CUDNN_DATA_HALF: {
      byte_size = 2;
      break;
    }
    case CUDNN_DATA_INT8:
    case CUDNN_DATA_UINT8: {
      byte_size = 1;
      break;
    }
#if CUDNN_VERSION > 7200
    case CUDNN_DATA_INT8x32: {
      byte_size = 32;
      break;
    }
#endif
    default: {
      UNIMPLEMENTED();
    }
  }
  return byte_size;
}

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
