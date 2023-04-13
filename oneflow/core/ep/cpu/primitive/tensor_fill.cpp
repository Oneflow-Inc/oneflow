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
#include "oneflow/core/ep/include/primitive/tensor_fill.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<typename T>
class TensorFillImpl : public TensorFill {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorFillImpl);
  TensorFillImpl() = default;
  ~TensorFillImpl() override = default;

  void Launch(Stream* stream, const void* src, void* dst, size_t count) override {
    const T* value = reinterpret_cast<const T*>(src);
    std::fill_n(reinterpret_cast<T*>(dst), count, value[0]);
  }
};

template<typename T>
std::unique_ptr<TensorFill> NewTensorFill() {
  return std::unique_ptr<TensorFill>(new TensorFillImpl<T>());
}

class TensorFillFactoryImpl : public TensorFillFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorFillFactoryImpl);
  TensorFillFactoryImpl() = default;
  ~TensorFillFactoryImpl() override = default;

  std::unique_ptr<TensorFill> New(DataType data_type) override {
#define MAKE_NEW_FILL_ENTRY(type_cpp, type_proto) {type_proto, NewTensorFill<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<TensorFill>()>> new_fill_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_FILL_ENTRY, CPU_PRIMITIVE_ALL_TYPE_SEQ)};
#undef MAKE_NEW_ADD_ENTRY
    const auto it = new_fill_handle.find(data_type);
    if (it != new_fill_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, TensorFillFactory, TensorFillFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
