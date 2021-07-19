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
#include "oneflow/core/device/rocm_util.h"
#include "oneflow/core/device/miopen_util.h"

namespace oneflow {

#ifdef WITH_ROCM

miopenDataType_t GetMiopenDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_miopen) \
  if (val == GetDataType<type_cpp>::value) { return type_miopen; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, MIOPEN_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNIMPLEMENTED();
}

MiopenTensorDesc::MiopenTensorDesc() { OF_MIOPEN_CHECK(miopenCreateTensorDescriptor(&val_)); }
MiopenTensorDesc::~MiopenTensorDesc() { OF_MIOPEN_CHECK(miopenDestroyTensorDescriptor(val_)); }
MiopenTensorDesc::MiopenTensorDesc(DataType data_type, int n, int c,
                                 int h, int w) {
  OF_MIOPEN_CHECK(miopenCreateTensorDescriptor(&val_));
  OF_MIOPEN_CHECK(miopenSet4dTensorDescriptor(val_, GetMiopenDataType(data_type), n, c, h, w));
}
MiopenTensorDesc::MiopenTensorDesc(DataType data_type, int dims, int* dim, int* stride) {
  OF_MIOPEN_CHECK(miopenCreateTensorDescriptor(&val_));
  OF_MIOPEN_CHECK(miopenSetTensorDescriptor(val_, GetMiopenDataType(data_type), dims, dim, stride));
}
MiopenTensorDesc::MiopenTensorDesc(DataType data_type, const ShapeView& shape) {
  OF_MIOPEN_CHECK(miopenCreateTensorDescriptor(&val_));

  if (shape.NumAxes() == 3) {
    int data_num = static_cast<int>(shape.At(0));
    int channels = static_cast<int>(shape.At(1));
    int kernel_h = static_cast<int>(shape.At(2));
    int kernel_w = 1;
    OF_MIOPEN_CHECK(miopenSet4dTensorDescriptor(val_, GetMiopenDataType(data_type),
                                              data_num, channels, kernel_h, kernel_w));
  } else if (shape.NumAxes() == 4) {
    int data_num = static_cast<int>(shape.At(0));
    int channels = static_cast<int>(shape.At(1));
    int kernel_h = static_cast<int>(shape.At(2));
    int kernel_w = static_cast<int>(shape.At(3));
    OF_MIOPEN_CHECK(miopenSet4dTensorDescriptor(val_, GetMiopenDataType(data_type),
                                              data_num, channels, kernel_h, kernel_w));
  } else {
    std::vector<int> tensor_dim({shape.ptr(), shape.ptr() + shape.NumAxes()});
    std::vector<int> stride_of_tensor(shape.NumAxes(), 1);
    for (int32_t i = shape.NumAxes() - 2; i >= 0; --i) {
      stride_of_tensor[i] = stride_of_tensor[i + 1] * shape.At(i + 1);
    }

    OF_MIOPEN_CHECK(miopenSetTensorDescriptor(val_, GetMiopenDataType(data_type), shape.NumAxes(),
                                              tensor_dim.data(), stride_of_tensor.data()));
  }
}

// MiopenFilterDesc::~MiopenFilterDesc() { OF_MIOPEN_CHECK(miopenDestroyTensorDescriptor(val_)); }

// MiopenFilterDesc::MiopenFilterDesc(DataType data_type, const ShapeView& shape) {
//   OF_MIOPEN_CHECK(miopenCreateTensorDescriptor(&val_));

//   if (shape.NumAxes() == 3) {
//     int filters = static_cast<int>(shape.At(0));
//     int c = static_cast<int>(shape.At(1));
//     int kernel_h = static_cast<int>(shape.At(2));
//     int kernel_w = 1;
//     OF_MIOPEN_CHECK(miopenSet4dTensorDescriptor(val_, GetMiopenDataType(data_type),
//                                               filters, c, kernel_h, kernel_w));
//   } else if (shape.NumAxes() == 4) {
//     int filters = static_cast<int>(shape.At(0));
//     int kernel_h = static_cast<int>(shape.At(2));
//     int kernel_w = static_cast<int>(shape.At(3));
//     int c = static_cast<int>(shape.At(1));
//     OF_MIOPEN_CHECK(miopenSet4dTensorDescriptor(val_, GetMiopenDataType(data_type),
//                                               filters, c, kernel_h, kernel_w));
//   } else {
//     std::vector<int> dims({shape.ptr(), shape.ptr() + shape.NumAxes()});
//     std::vector<int> stride_of_tensor(shape.NumAxes(), 1);
//     for (int32_t i = shape.NumAxes() - 2; i >= 0; --i) {
//       stride_of_tensor[i] = stride_of_tensor[i + 1] * shape.At(i + 1);
//     }
//     OF_MIOPEN_CHECK(miopenSetTensorDescriptor(val_, GetMiopenDataType(data_type),
//                                               dims.size(), dims.data(), stride_of_tensor.data()));
//   }
// }

// MiopenActivationDesc::MiopenActivationDesc(miopenActivationMode_t mode,
//                                            miopenNanPropagation_t relu_nan_opt, double coef) {
//   OF_MIOPEN_CHECK(miopenCreateActivationDescriptor(&val_));
//   OF_MIOPEN_CHECK(miopenSetActivationDescriptor(val_, mode, relu_nan_opt, coef));
// }

// MiopenActivationDesc::~MiopenActivationDesc() {
//   OF_MIOPEN_CHECK(miopenDestroyActivationDescriptor(val_));
// }

size_t GetMiopenDataTypeByteSize(miopenDataType_t data_type) {
  size_t byte_size = 0;
  switch (data_type) {
    case miopenFloat:
    case miopenInt32:
    case miopenInt8x4: {
      byte_size = 4;
      break;
    }
    case miopenHalf: {
      byte_size = 2;
      break;
    }
    case miopenInt8: {
      byte_size = 1;
      break;
    }
    default: {
      UNIMPLEMENTED();
    }
  }
  return byte_size;
}

template<typename T>
void* MiopenSPOnePtr() {
  static float fval = 1.0f;
  static double dval = 1.0;
  void* ret = std::is_same<T, double>::value ? static_cast<void*>(&dval)
                                                   : static_cast<void*>(&fval);
  return ret;
}

template<typename T>
void* MiopenSPZeroPtr() {
  static float fval = 0.0f;
  static double dval = 0.0;
  void* ret = std::is_same<T, double>::value ? static_cast<void*>(&dval)
                                                   : static_cast<void*>(&fval);
  return ret;
}

template void* MiopenSPOnePtr<float>();
template void* MiopenSPOnePtr<double>();
template void* MiopenSPOnePtr<float16>();

template void* MiopenSPZeroPtr<float>();
template void* MiopenSPZeroPtr<double>();
template void* MiopenSPZeroPtr<float16>();

#endif  // WITH_ROCM

}  // namespace oneflow
