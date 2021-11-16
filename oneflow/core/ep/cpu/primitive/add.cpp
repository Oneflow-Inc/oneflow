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
#include "oneflow/core/stream/cpu/cpu_stream_context.h"

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

template<typename T, dnnl::memory::data_type type_onednn, CalculateType type_calculate,
         typename Enable = void>
class AddImpl;

template<typename T, dnnl::memory::data_type type_onednn, CalculateType type_calculate>
class AddImpl<T, type_onednn, type_calculate,
              typename std::enable_if<type_calculate == CalculateType::Default>::type>
    : public Add {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddImpl);
  AddImpl() = default;
  ~AddImpl() override = default;

  using Add::Launch;
  void Launch(StreamContext* stream_ctx, const void* const* srcs, size_t arity, void* dst,
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

template<typename T, dnnl::memory::data_type type_onednn, CalculateType type_calculate>
class AddImpl<T, type_onednn, type_calculate,
              typename std::enable_if<type_calculate == CalculateType::oneDNN>::type> : public Add {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddImpl);
  AddImpl() = default;
  ~AddImpl() override = default;

  using Add::Launch;
  void Launch(StreamContext* stream_ctx, const void* const* srcs, size_t arity, void* dst,
              size_t count) override {
    dnnl::engine* onednn_engine =
        CHECK_NOTNULL(dynamic_cast<CpuStreamContext*>(stream_ctx))->onednn_engine();
    dnnl::stream* onednn_stream =
        CHECK_NOTNULL(dynamic_cast<CpuStreamContext*>(stream_ctx))->onednn_stream();

    dnnl::memory::dims src_dims = {(dnnl::memory::dim)count};
    std::vector<dnnl::memory::desc> src_md;
    std::vector<dnnl::memory> src_mem;
    src_md.reserve(arity);
    src_mem.reserve(arity);

    for (int i = 0; i < arity; i++) {
      auto md = dnnl::memory::desc(src_dims, type_onednn, dnnl::memory::format_tag::x);
      auto mem = dnnl::memory(md, *onednn_engine, (void*)(srcs)[i]);
      src_md.emplace_back(md);
      src_mem.emplace_back(mem);
    }

    std::vector<float> scales(arity, 1.0);

    auto sum_pd = dnnl::sum::primitive_desc(scales, src_md, *onednn_engine);

    auto sum_prim = dnnl::sum(sum_pd);

    auto dst_mem = dnnl::memory(sum_pd.dst_desc(), *onednn_engine, dst);

    std::unordered_map<int, dnnl::memory> sum_args{{DNNL_ARG_DST, dst_mem}};
    for (int n = 0; n < arity; ++n) { sum_args.insert({DNNL_ARG_MULTIPLE_SRC + n, src_mem[n]}); }

    sum_prim.execute(*onednn_stream, sum_args);
    onednn_stream->wait();
  }
};

template<typename T, dnnl::memory::data_type type_onednn, CalculateType type_calculate>
std::unique_ptr<Add> NewAdd() {
  return std::unique_ptr<Add>(new AddImpl<T, type_onednn, type_calculate>());
}

class AddFactoryImpl : public AddFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddFactoryImpl);
  AddFactoryImpl() = default;
  ~AddFactoryImpl() override = default;

  std::unique_ptr<Add> New(DataType data_type) override {
#define MAKE_NEW_ADD_ENTRY(type_cpp, type_proto, type_onednn, type_calculate) \
  {type_proto, NewAdd<type_cpp, type_onednn, type_calculate>},

    static const std::map<DataType, std::function<std::unique_ptr<Add>()>> new_add_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_ADD_ENTRY, CPU_PRIMITIVE_ONEDNN_NATIVE_TYPE_SEQ)};
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
