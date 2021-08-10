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
#include "half.hpp"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/device_register_cpu.h"

namespace oneflow {

namespace {

template<typename T>
std::size_t GetDataTypeBytes() {
  return sizeof(T);
}

#define MAKE_DATA_TYPE_BYTES_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(std::size_t, GetDataTypeBytes, MAKE_DATA_TYPE_BYTES_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ));

class DTypeMeta final {
 public:
  DTypeMeta(const std::string& name, bool is_signed, bool is_floating_point, bool is_complex)
      : name_(name),
        is_signed_(is_signed),
        is_floating_point_(is_floating_point),
        is_complex_(is_complex) {}
  DTypeMeta(const DTypeMeta&) = default;
  DTypeMeta(DTypeMeta&) = default;
  ~DTypeMeta() = default;

  const std::string& name() const { return name_; }
  bool is_signed() const { return is_signed_; }
  bool is_floating_point() const { return is_floating_point_; }
  bool is_complex() const { return is_complex_; }

 private:
  const std::string name_;
  const bool is_signed_;
  const bool is_floating_point_;
  const bool is_complex_;
};

Maybe<const DTypeMeta&> DTypeMeta4DataType(DataType data_type) {
  static HashMap<DataType, DTypeMeta> data_type2dtype_meta{
      {DataType::kInvalidDataType, DTypeMeta("oneflow.invalid_data_type", false, false, false)},
      {DataType::kChar, DTypeMeta("oneflow.char", false, false, false)},
      {DataType::kFloat16, DTypeMeta("oneflow.float16", true, true, false)},
      {DataType::kFloat, DTypeMeta("oneflow.float32", true, true, false)},
      {DataType::kDouble, DTypeMeta("oneflow.float64", true, true, false)},
      {DataType::kInt8, DTypeMeta("oneflow.int8", true, false, false)},
      {DataType::kInt32, DTypeMeta("oneflow.int32", true, false, false)},
      {DataType::kInt64, DTypeMeta("oneflow.int64", true, false, false)},
      {DataType::kUInt8, DTypeMeta("oneflow.uint8", false, false, false)},
      {DataType::kOFRecord, DTypeMeta("oneflow.of_record", false, false, false)},
      {DataType::kTensorBuffer, DTypeMeta("oneflow.tensor_buffer", false, false, false)},
  };
  return MapAt(data_type2dtype_meta, data_type);
};

}  // namespace

Maybe<const Symbol<DType>&> DType::Get(DataType data_type) {
  static HashMap<DataType, const Symbol<DType>> data_type2dtype{
#define MAKE_ENTRY(data_type) {OF_PP_CAT(DataType::k, data_type), data_type()},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, DTYPE_SEQ)
#undef MAKE_ENTRY
  };
  return MapAt(data_type2dtype, data_type);
}

Maybe<size_t> DType::bytes() const {
  // DataType::OFRecord and DataType::TensorBuffer don't have fixed byte size
  if (data_type() == DataType::kInvalidDataType || data_type() == DataType::kOFRecord
      || data_type() == DataType::kTensorBuffer) {
    OF_UNIMPLEMENTED();
  }
  return SwitchGetDataTypeBytes(SwitchCase(data_type()));
}

bool DType::is_signed() const { return CHECK_JUST(DTypeMeta4DataType(data_type_)).is_signed(); }

bool DType::is_complex() const { return CHECK_JUST(DTypeMeta4DataType(data_type_)).is_complex(); }

bool DType::is_floating_point() const {
  return CHECK_JUST(DTypeMeta4DataType(data_type_)).is_floating_point();
}

const std::string& DType::name() const { return CHECK_JUST(DTypeMeta4DataType(data_type_)).name(); }

#define DEFINE_GET_DATA_TYPE_FUNCTION(data_type)                                   \
  const Symbol<DType>& DType::data_type() {                                        \
    static const auto& dtype = SymbolOf(DType(OF_PP_CAT(DataType::k, data_type))); \
    return dtype;                                                                  \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_GET_DATA_TYPE_FUNCTION, DTYPE_SEQ)
#undef DEFINE_GET_DATA_TYPE_FUNCTION

}  // namespace oneflow
