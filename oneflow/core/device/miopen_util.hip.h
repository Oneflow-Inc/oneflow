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
#ifndef ONEFLOW_CORE_DEVICE_MIOPEN_UTIL_H_
#define ONEFLOW_CORE_DEVICE_MIOPEN_UTIL_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"

#ifdef WITH_HIP

#include <miopen/miopen.h>

namespace oneflow {

#define MIOPEN_DATA_TYPE_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(float, miopenFloat)   \
  OF_PP_MAKE_TUPLE_SEQ(float16, miopenHalf)  \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, miopenInt8)   \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, miopenInt32)

miopenDataType_t GetMiopenDataType(DataType);

template<typename T>
struct MiopenDataType;

#define SPECIALIZE_MIOPEN_DATA_TYPE(type_cpp, type_miopen) \
  template<>                                             \
  struct MiopenDataType<type_cpp> : std::integral_constant<miopenDataType_t, type_miopen> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_MIOPEN_DATA_TYPE, MIOPEN_DATA_TYPE_SEQ);
#undef SPECIALIZE_MIOPEN_DATA_TYPE

class MiopenTensorDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MiopenTensorDesc);
  MiopenTensorDesc();
  ~MiopenTensorDesc();

  MiopenTensorDesc(DataType, int n, int c, int h, int w);
  MiopenTensorDesc(DataType data_type, int dims, int* dim, int* stride);
  MiopenTensorDesc(DataType data_type, const ShapeView& shape);

  const miopenTensorDescriptor_t& Get() const { return val_; }

 private:
  miopenTensorDescriptor_t val_;
};

// class MiopenFilterDesc final {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(MiopenFilterDesc);
//   MiopenFilterDesc() = delete;
//   ~MiopenFilterDesc();

//   MiopenFilterDesc(DataType data_type, const ShapeView& shape);

//   const TensorDescriptor& Get() const { return val_; }

//  private:
//   TensorDescriptor val_;
// };

// class MiopenActivationDesc final {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(MiopenActivationDesc);
//   MiopenActivationDesc() = delete;
//   ~MiopenActivationDesc();

//   MiopenActivationDesc(miopenActivationMode_t mode, miopenNanPropagation_t relu_nan_opt, double coef);

//   const miopenActivationDescriptor_t& Get() const { return val_; }

//  private:
//   miopenActivationDescriptor_t val_;
// };

size_t GetMiopenDataTypeByteSize(miopenDataType_t data_type);

// SP for scaling parameter
template<typename T>
void* MiopenSPOnePtr();

template<typename T>
void* MiopenSPZeroPtr();

}  // namespace oneflow

#endif  // WITH_HIP

#endif  // ONEFLOW_CORE_DEVICE_MIOPEN_UTIL_H_
