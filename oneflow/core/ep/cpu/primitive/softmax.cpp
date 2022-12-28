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
#include "oneflow/core/ep/include/primitive/softmax.h"
#include "oneflow/core/ep/include/primitive/log_softmax.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include "oneflow/core/ep/common/primitive/util.h"
#include "oneflow/core/ep/common/onednn.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

enum class Algorithm {
  kSoftmax,
  kLogSoftmax,
};

template<Algorithm algorithm, typename T>
void SoftmaxCpu(size_t rows, size_t cols, const T* x, T* y) {
  for (size_t i = 0; i < rows; ++i) {
    size_t row_offset = i * cols;
    const T* row_x = x + row_offset;
    T* row_y = y + row_offset;
    const T row_max = *std::max_element(row_x, row_x + cols);
    T row_sum = 0;
    for (size_t j = 0; j < cols; ++j) {
      if (algorithm == Algorithm::kSoftmax) {
        T exp_x = std::exp(row_x[j] - row_max);
        row_sum += exp_x;
        row_y[j] = exp_x;
      } else if (algorithm == Algorithm::kLogSoftmax) {
        row_y[j] = row_x[j] - row_max;
        row_sum += std::exp(row_y[j]);
      } else {
        UNIMPLEMENTED();
      }
    }
    for (size_t j = 0; j < cols; ++j) {
      if (algorithm == Algorithm::kSoftmax) {
        row_y[j] /= row_sum;
      } else if (algorithm == Algorithm::kLogSoftmax) {
        row_y[j] -= std::log(row_sum);
      } else {
        UNIMPLEMENTED();
      }
    }
  }
}

template<typename SoftmaxBase, Algorithm algorithm, typename T>
class SoftmaxImpl : public SoftmaxBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxImpl);
  SoftmaxImpl() = default;
  ~SoftmaxImpl() override = default;

  void Launch(Stream* stream, size_t rows, size_t cols, const void* x, void* y) override {
    SoftmaxCpu<algorithm, T>(rows, cols, reinterpret_cast<const T*>(x), reinterpret_cast<T*>(y));
  }
};

#ifdef WITH_ONEDNN

template<class OneDnnSoftmax, dnnl::memory::data_type data_type>
void SoftmaxOneDnn(Stream* stream, size_t rows, size_t cols, const void* x, void* y) {
  stream->As<CpuStream>()->onednn_executor()->Launch(
      [&](dnnl::engine* onednn_engine, dnnl::stream* onednn_stream) {
        dnnl::memory::dims src_dims = {static_cast<dnnl::memory::dim>(rows),
                                       static_cast<dnnl::memory::dim>(cols)};

        auto src_md = dnnl::memory::desc(src_dims, data_type, dnnl::memory::format_tag::nc);
        auto src_mem = dnnl::memory(src_md, *onednn_engine, const_cast<void*>(x));
        auto dst_mem = dnnl::memory(src_md, *onednn_engine, y);
        auto softmax_d = typename OneDnnSoftmax::desc(dnnl::prop_kind::forward, src_md, 1);
        auto softmax_pd = typename OneDnnSoftmax::primitive_desc(softmax_d, *onednn_engine);
        auto softmax_prim = OneDnnSoftmax(softmax_pd);

        softmax_prim.execute(*onednn_stream, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
      });
}

template<typename SoftmaxBase, Algorithm algorithm, dnnl::memory::data_type data_type>
class OneDnnSoftmaxImpl;

#define CPU_PRIMITIVE_SOFTMAX_ONEDNN_IMPL(oneflow_algorithm, onednn_algorithm)               \
  template<typename SoftmaxBase, dnnl::memory::data_type data_type>                          \
  class OneDnnSoftmaxImpl<SoftmaxBase, oneflow_algorithm, data_type> : public SoftmaxBase {  \
   public:                                                                                   \
    OF_DISALLOW_COPY_AND_MOVE(OneDnnSoftmaxImpl);                                            \
    OneDnnSoftmaxImpl() = default;                                                           \
    ~OneDnnSoftmaxImpl() override = default;                                                 \
                                                                                             \
    using OneDnnClass = onednn_algorithm;                                                    \
    void Launch(Stream* stream, size_t rows, size_t cols, const void* x, void* y) override { \
      SoftmaxOneDnn<OneDnnClass, data_type>(stream, rows, cols, x, y);                       \
    }                                                                                        \
  }

CPU_PRIMITIVE_SOFTMAX_ONEDNN_IMPL(Algorithm::kSoftmax, dnnl::softmax_forward);
CPU_PRIMITIVE_SOFTMAX_ONEDNN_IMPL(Algorithm::kLogSoftmax, dnnl::logsoftmax_forward);
#undef CPU_PRIMITIVE_SOFTMAX_ONEDNN_IMPL

template<typename SoftmaxBase, Algorithm algorithm, dnnl::memory::data_type data_type>
std::unique_ptr<SoftmaxBase> NewOneDnnSoftmax() {
  return std::unique_ptr<SoftmaxBase>(new OneDnnSoftmaxImpl<SoftmaxBase, algorithm, data_type>());
}

#endif  // WITH_ONEDNN

template<typename SoftmaxBase, Algorithm algorithm, typename T>
std::unique_ptr<SoftmaxBase> NewSoftmax() {
  return std::unique_ptr<SoftmaxBase>(new SoftmaxImpl<SoftmaxBase, algorithm, T>());
}

template<typename FactoryBase, typename SoftmaxBase, Algorithm algorithm>
class GenericSoftmaxFactoryImpl : public FactoryBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GenericSoftmaxFactoryImpl);
  GenericSoftmaxFactoryImpl() = default;
  ~GenericSoftmaxFactoryImpl() override = default;

  std::unique_ptr<SoftmaxBase> New(DataType data_type) override {
#define MAKE_NEW_SOFTMAX_ENTRY(type_cpp, type_proto) \
  {type_proto, NewSoftmax<SoftmaxBase, algorithm, type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<SoftmaxBase>()>>
        new_softmax_handle{
            OF_PP_FOR_EACH_TUPLE(MAKE_NEW_SOFTMAX_ENTRY, CPU_PRIMITIVE_FLOATING_TYPE_SEQ)};

#undef MAKE_NEW_SOFTMAX_ENTRY

#ifdef WITH_ONEDNN

    if (OneDnnIsEnabled() && data_type == DataType::kFloat) {
      static std::function<std::unique_ptr<SoftmaxBase>()> onednn_softmax =
          NewOneDnnSoftmax<SoftmaxBase, algorithm, dnnl::memory::data_type::f32>;
      return onednn_softmax();
    }

#endif
    return NewPrimitiveFromHandlers(new_softmax_handle, data_type);
  }
};

using SoftmaxFactoryImpl = GenericSoftmaxFactoryImpl<SoftmaxFactory, Softmax, Algorithm::kSoftmax>;
using LogSoftmaxFactoryImpl =
    GenericSoftmaxFactoryImpl<LogSoftmaxFactory, LogSoftmax, Algorithm::kLogSoftmax>;
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, SoftmaxFactory, SoftmaxFactoryImpl);
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, LogSoftmaxFactory, LogSoftmaxFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
