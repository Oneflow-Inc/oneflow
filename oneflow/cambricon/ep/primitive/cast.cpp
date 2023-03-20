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
#include "oneflow/cambricon/ep/primitive/cast.h"

#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

namespace ep {
namespace primitive {

cnnlCastDataType_t GetCnnlCastType(DataType from, DataType to) {
  static std::map<std::pair<DataType, DataType>, cnnlCastDataType_t> cast_dtype_table = {
      {{kFloat, kFloat16}, CNNL_CAST_FLOAT_TO_HALF},
      {{kFloat, kInt32}, CNNL_CAST_FLOAT_TO_INT32},
      {{kFloat, kInt8}, CNNL_CAST_FLOAT_TO_INT8},
      {{kFloat, kUInt8}, CNNL_CAST_FLOAT_TO_UINT8},
      {{kFloat16, kFloat}, CNNL_CAST_HALF_TO_FLOAT},
      {{kFloat16, kInt32}, CNNL_CAST_HALF_TO_INT32},
      {{kFloat16, kInt8}, CNNL_CAST_HALF_TO_INT8},
      {{kInt32, kInt8}, CNNL_CAST_INT32_TO_INT8},
      {{kFloat16, kBool}, CNNL_CAST_HALF_TO_BOOL},
      {{kInt8, kFloat}, CNNL_CAST_INT8_TO_FLOAT},
      {{kInt8, kFloat16}, CNNL_CAST_INT8_TO_HALF},
      {{kInt8, kInt32}, CNNL_CAST_INT8_TO_INT32},
      {{kUInt8, kFloat}, CNNL_CAST_UINT8_TO_FLOAT},
      {{kUInt8, kFloat16}, CNNL_CAST_UINT8_TO_HALF},
      {{kBool, kFloat}, CNNL_CAST_BOOL_TO_FLOAT},
      {{kBool, kFloat16}, CNNL_CAST_BOOL_TO_HALF},
      {{kBool, kInt32}, CNNL_CAST_BOOL_TO_INT32},
      {{kUInt8, kInt32}, CNNL_CAST_UINT8_TO_INT32},
      {{kInt32, kInt64}, CNNL_CAST_INT32_TO_INT64},
      {{kInt64, kInt32}, CNNL_CAST_INT64_TO_INT32},
      {{kInt32, kBool}, CNNL_CAST_INT32_TO_BOOL},
      {{kUInt8, kInt64}, CNNL_CAST_UINT8_TO_INT64},
      {{kUInt64, kUInt32}, CNNL_CAST_UINT64_TO_UINT32},
      {{kInt64, kUInt32}, CNNL_CAST_INT64_TO_UINT32},
      {{kInt64, kFloat}, CNNL_CAST_INT64_TO_FLOAT},
      {{kInt64, kFloat16}, CNNL_CAST_INT64_TO_HALF},
      {{kFloat, kInt64}, CNNL_CAST_FLOAT_TO_INT64},
      {{kFloat16, kInt64}, CNNL_CAST_HALF_TO_INT64},
      {{kInt32, kFloat}, CNNL_CAST_INT32_TO_FLOAT},
  };
  auto it = cast_dtype_table.find(std::make_pair(from, to));
  CHECK_OR_THROW(it != cast_dtype_table.end())
      << "cambricon cnnl does not support cast tensor from type " << DataType_Name(from) << " to "
      << DataType_Name(to);
  return it->second;
}

namespace {

class CastImpl : public Cast {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastImpl);
  CastImpl(DataType from, DataType to)
      : from_type_(ConvertToCnnlDataType(from)),
        to_type_(ConvertToCnnlDataType(to)),
        cnnl_cast_type_(GetCnnlCastType(from, to)) {}
  ~CastImpl() override = default;

  void Launch(Stream* stream, const void* from, void* to, size_t count) override {
    CnnlTensorDescriptor from_desc, to_desc;
    int64_t shape[1] = {static_cast<int64_t>(count)};
    from_desc.set(1, shape, from_type_);
    to_desc.set(1, shape, to_type_);
    OF_CNNL_CHECK(cnnlCastDataType(stream->As<ep::MluStream>()->cnnl_handle(), from_desc.desc(),
                                   from, cnnl_cast_type_, to_desc.desc(), to));
  }

 private:
  cnnlDataType_t from_type_;
  cnnlDataType_t to_type_;
  cnnlCastDataType_t cnnl_cast_type_;
};

class CastFactoryImpl : public CastFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastFactoryImpl);
  CastFactoryImpl() = default;
  ~CastFactoryImpl() override = default;

  std::unique_ptr<Cast> New(DataType from, DataType to) override {
    return std::unique_ptr<Cast>(new CastImpl(from, to));
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kMLU, CastFactory, CastFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
