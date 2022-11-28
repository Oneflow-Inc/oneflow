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
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/common/primitive/util.h"
#include "oneflow/core/ep/common/onednn.h"

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
class AddDefaultImpl : public Add {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddDefaultImpl);
  AddDefaultImpl() = default;
  ~AddDefaultImpl() override = default;

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

#ifdef WITH_ONEDNN
class AddOneDnnImpl : public Add {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddOneDnnImpl);
  explicit AddOneDnnImpl(dnnl::memory::data_type type) : type_onednn_(type){};
  ~AddOneDnnImpl() override = default;

  using Add::Launch;
  void Launch(Stream* stream, const void* const* srcs, size_t arity, void* dst,
              size_t count) override {
    if (arity < 2) {
      // TODO: arity 0 and 1
      UNIMPLEMENTED() << "Addn only supports summation of 2 or more tensors";
    } else if (arity == 2) {
      if (srcs[1] == dst && srcs[0] != dst) {
        LOG(FATAL) << "Only the first parameter can be operated inplace";
      }
    } else {
      for (int i = 2; i < arity; i++) {
        if (srcs[i] == dst) { LOG(FATAL) << "Only the first parameter can be operated inplace"; }
      }
    }

    stream->As<CpuStream>()->onednn_executor()->Launch(
        [&](dnnl::engine* onednn_engine, dnnl::stream* onednn_stream) {
          dnnl::memory::dims src_dims = {static_cast<dnnl::memory::dim>(count)};
          std::vector<dnnl::memory::desc> src_md;
          std::vector<dnnl::memory> src_mem;
          src_md.reserve(arity);
          src_mem.reserve(arity);

          for (int i = 0; i < arity; i++) {
            auto md = dnnl::memory::desc(src_dims, type_onednn_, dnnl::memory::format_tag::x);
            auto mem = dnnl::memory(md, *onednn_engine, (void*)(srcs)[i]);
            src_md.emplace_back(md);
            src_mem.emplace_back(mem);
          }

          std::vector<float> scales(arity, 1.0);
          auto sum_pd = dnnl::sum::primitive_desc(scales, src_md, *onednn_engine);
          auto sum_prim = dnnl::sum(sum_pd);
          auto dst_mem = dnnl::memory(sum_pd.dst_desc(), *onednn_engine, dst);
          std::unordered_map<int, dnnl::memory> sum_args{{DNNL_ARG_DST, dst_mem}};
          for (int i = 0; i < arity; ++i) {
            sum_args.insert({DNNL_ARG_MULTIPLE_SRC + i, src_mem[i]});
          }

          sum_prim.execute(*onednn_stream, sum_args);
        });
  }

 private:
  dnnl::memory::data_type type_onednn_;
};

#endif

template<typename T>
std::unique_ptr<Add> NewAdd() {
  return std::unique_ptr<Add>(new AddDefaultImpl<T>());
}

#ifdef WITH_ONEDNN

template<dnnl::memory::data_type type_onednn>
std::unique_ptr<Add> NewOneDnnAdd() {
  return std::unique_ptr<Add>(new AddOneDnnImpl(type_onednn));
}

#endif

#define CPU_PRIMITIVE_ADD_ONEDNN_TYPE_SEQ \
  CPU_PRIMITIVE_ONEDNN_INT8_TYPE_SEQ      \
  CPU_PRIMITIVE_ONEDNN_UINT8_TYPE_SEQ     \
  CPU_PRIMITIVE_ONEDNN_INT32_TYPE_SEQ     \
  CPU_PRIMITIVE_ONEDNN_FLOAT_TYPE_SEQ     \
  CPU_PRIMITIVE_ONEDNN_FLOAT16_TYPE_SEQ   \
  CPU_PRIMITIVE_ONEDNN_BFLOAT16_TYPE_SEQ

#define CPU_PRIMITIVE_ADD_DEFAULT_TYPE_SEQ \
  CPU_PRIMITIVE_BOOL_TYPE_SEQ              \
  CPU_PRIMITIVE_CHAR_TYPE_SEQ              \
  CPU_PRIMITIVE_DOUBLE_TYPE_SEQ            \
  CPU_PRIMITIVE_INT64_TYPE_SEQ

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
#ifdef WITH_ONEDNN

#define MAKE_NEW_ONEDNN_ADD_ENTRY(type_onednn, type_proto) {type_proto, NewOneDnnAdd<type_onednn>},

    static const std::map<DataType, std::function<std::unique_ptr<Add>()>> new_add_onednn_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_ONEDNN_ADD_ENTRY, CPU_PRIMITIVE_ADD_ONEDNN_TYPE_SEQ)};

#undef MAKE_NEW_ONEDNN_ADD_ENTRY

    if (OneDnnIsEnabled()) {
      auto add_primitive = NewPrimitiveFromHandlers(new_add_onednn_handle, data_type);
      if (add_primitive) { return add_primitive; }
    }

#endif
    return NewPrimitiveFromHandlers(new_add_handle, data_type);
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, AddFactory, AddFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
