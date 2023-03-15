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
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<typename T>
T GetValue(Scalar value) {
  return value.Value<T>();
}

template<>
float16 GetValue<float16>(Scalar value) {
  return static_cast<float16>(GetValue<float>(value));
}

template<>
bfloat16 GetValue<bfloat16>(Scalar value) {
  return static_cast<bfloat16>(GetValue<float>(value));
}

template<typename T>
class FillImpl : public Fill {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FillImpl);
  FillImpl() = default;
  ~FillImpl() override = default;

  void Launch(Stream* stream, void* dst, Scalar value, size_t count) override {
    std::fill_n(reinterpret_cast<T*>(dst), count, GetValue<T>(value));
  }
};

template<typename T>
std::unique_ptr<Fill> NewFill() {
  return std::unique_ptr<Fill>(new FillImpl<T>());
}

class FillFactoryImpl : public FillFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FillFactoryImpl);
  FillFactoryImpl() = default;
  ~FillFactoryImpl() override = default;

  std::unique_ptr<Fill> New(DataType data_type) override {
#define MAKE_NEW_FILL_ENTRY(type_cpp, type_proto) {type_proto, NewFill<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<Fill>()>> new_fill_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_FILL_ENTRY,
                             CPU_PRIMITIVE_ALL_TYPE_SEQ CPU_PRIMITIVE_COMPLEX_TYPE_SEQ)};
#undef MAKE_NEW_ADD_ENTRY
    const auto it = new_fill_handle.find(data_type);
    if (it != new_fill_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, FillFactory, FillFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
