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
#include "oneflow/core/ep/include/primitive/add.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<typename T, size_t arity>
void AddCpu(const T* const* srcs, T* dst, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    T sum = T(0);
    for (size_t a = 0; a < arity; ++a) { sum += srcs[a][i]; }
    dst[i] = sum;
  }
}

template<typename T>
void AddCpu(const T* const* srcs, size_t arity, T* dst, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    T sum = T(0);
    for (size_t a = 0; a < arity; ++a) { sum += srcs[a][i]; }
    dst[i] = sum;
  }
}

template<typename T>
class AddImpl : public Add {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddImpl);
  AddImpl() = default;
  ~AddImpl() override = default;

  using Add::Launch;
  void Launch(Stream* stream, const void* const* srcs, size_t arity, void* dst,
              size_t count) override {
#define ONE_IF(a)                                                                            \
  if (arity == a) {                                                                          \
    AddCpu<T, a>(reinterpret_cast<const T* const*>(srcs), reinterpret_cast<T*>(dst), count); \
  }
#define ONE_ELIF(a) else ONE_IF(a)
#define ONE_ELSE                                                                                 \
  else {                                                                                         \
    AddCpu<T>(reinterpret_cast<const T* const*>(srcs), arity, reinterpret_cast<T*>(dst), count); \
  }
    ONE_IF(0)
    ONE_ELIF(1)
    ONE_ELIF(2)
    ONE_ELIF(3)
    ONE_ELIF(4)
    ONE_ELIF(5)
    ONE_ELIF(6)
    ONE_ELIF(7)
    ONE_ELIF(8)
    ONE_ELSE
  }
};

template<typename T>
std::unique_ptr<Add> NewAdd() {
  return std::unique_ptr<Add>(new AddImpl<T>());
}

class AddFactoryImpl : public AddFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddFactoryImpl);
  AddFactoryImpl() = default;
  ~AddFactoryImpl() override = default;

  std::unique_ptr<Add> New(DataType data_type) override {
#define MAKE_NEW_ADD_ENTRY(type_cpp, type_proto) {type_proto, NewAdd<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<Add>()>> new_add_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_ADD_ENTRY, CPU_PRIMITIVE_ALL_TYPE_SEQ)};
#undef MAKE_NEW_ADD_ENTRY
    const auto it = new_add_handle.find(data_type);
    if (it != new_add_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, AddFactory, AddFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
