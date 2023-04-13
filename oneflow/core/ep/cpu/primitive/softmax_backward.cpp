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
#include "oneflow/core/ep/include/primitive/softmax_backward.h"
#include "oneflow/core/ep/include/primitive/log_softmax_backward.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include "oneflow/core/ep/common/onednn.h"
#include "oneflow/core/ep/common/primitive/util.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

enum class Algorithm {
  kSoftmax,
  kLogSoftmax,
};

template<Algorithm algorithm, typename T>
void SoftmaxBackwardCpu(size_t rows, size_t cols, const T* y, const T* dy, T* dx) {
  for (size_t i = 0; i < rows; ++i) {
    size_t row_offset = i * cols;
    const T* row_y = y + row_offset;
    const T* row_dy = dy + row_offset;
    T* row_dx = dx + row_offset;
    T row_sum = 0;
    for (size_t j = 0; j < cols; ++j) {
      if (algorithm == Algorithm::kSoftmax) {
        row_sum += row_y[j] * row_dy[j];
      } else if (algorithm == Algorithm::kLogSoftmax) {
        row_sum += row_dy[j];
      } else {
        UNIMPLEMENTED();
      }
    }
    for (size_t j = 0; j < cols; ++j) {
      if (algorithm == Algorithm::kSoftmax) {
        row_dx[j] = (row_dy[j] - row_sum) * row_y[j];
      } else if (algorithm == Algorithm::kLogSoftmax) {
        row_dx[j] = row_dy[j] - std::exp(row_y[j]) * row_sum;
      } else {
        UNIMPLEMENTED();
      }
    }
  }
}

template<typename SoftmaxBackwardBase, Algorithm algorithm, typename T>
class SoftmaxBackwardImpl : public SoftmaxBackwardBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxBackwardImpl);
  SoftmaxBackwardImpl() = default;
  ~SoftmaxBackwardImpl() override = default;

  void Launch(Stream* stream, size_t rows, size_t cols, const void* y, const void* dy,
              void* dx) override {
    SoftmaxBackwardCpu<algorithm, T>(rows, cols, reinterpret_cast<const T*>(y),
                                     reinterpret_cast<const T*>(dy), reinterpret_cast<T*>(dx));
  }
};

#ifdef WITH_ONEDNN

template<class OneDnnSoftmaxBackward, class OneDnnSoftmaxForward, dnnl::memory::data_type data_type>
void SoftmaxBackwardOneDnn(Stream* stream, size_t rows, size_t cols, const void* y, const void* dy,
                           void* dx) {
  stream->As<CpuStream>()->onednn_executor()->Launch([&](dnnl::engine* onednn_engine,
                                                         dnnl::stream* onednn_stream) {
    dnnl::memory::dims src_dims = {static_cast<dnnl::memory::dim>(rows),
                                   static_cast<dnnl::memory::dim>(cols)};
    // Input and output parameters of the same data type
    auto same_md = dnnl::memory::desc(src_dims, data_type, dnnl::memory::format_tag::nc);
    // Backward memory
    auto dst_mem = dnnl::memory(same_md, *onednn_engine, const_cast<void*>(y));
    auto diff_dst_mem = dnnl::memory(same_md, *onednn_engine, const_cast<void*>(dy));
    // Forward primitive description
    auto forward_desc = typename OneDnnSoftmaxForward::desc(dnnl::prop_kind::forward, same_md, 1);
    auto forward_prim_desc =
        typename OneDnnSoftmaxForward::primitive_desc(forward_desc, *onednn_engine);
    // Backward primitive description
    auto diff_src_mem = dnnl::memory(same_md, *onednn_engine, dx);
    auto backward_desc = typename OneDnnSoftmaxBackward::desc(same_md, same_md, 1);
    auto backward_prim_desc = typename OneDnnSoftmaxBackward::primitive_desc(
        backward_desc, *onednn_engine, forward_prim_desc);
    auto backward_prim = OneDnnSoftmaxBackward(backward_prim_desc);

    backward_prim.execute(*onednn_stream, {{DNNL_ARG_DIFF_DST, diff_dst_mem},
                                           {DNNL_ARG_DST, dst_mem},
                                           {DNNL_ARG_DIFF_SRC, diff_src_mem}});
  });
}

template<typename SoftmaxBackwardBase, Algorithm algorithm, dnnl::memory::data_type data_type>
class OneDnnSoftmaxBackwardImpl;

#define CPU_PRIMITIVE_SOFTMAX_ONEDNN_IMPL(oneflow_algorithm, onednn_backward_algorithm,      \
                                          onednn_forward_algorithm)                          \
  template<typename SoftmaxBackwardBase, dnnl::memory::data_type data_type>                  \
  class OneDnnSoftmaxBackwardImpl<SoftmaxBackwardBase, oneflow_algorithm, data_type>         \
      : public SoftmaxBackwardBase {                                                         \
   public:                                                                                   \
    OF_DISALLOW_COPY_AND_MOVE(OneDnnSoftmaxBackwardImpl);                                    \
    OneDnnSoftmaxBackwardImpl() = default;                                                   \
    ~OneDnnSoftmaxBackwardImpl() override = default;                                         \
                                                                                             \
    void Launch(Stream* stream, size_t rows, size_t cols, const void* y, const void* dy,     \
                void* dx) override {                                                         \
      SoftmaxBackwardOneDnn<onednn_backward_algorithm, onednn_forward_algorithm, data_type>( \
          stream, rows, cols, y, dy, dx);                                                    \
    }                                                                                        \
  }

CPU_PRIMITIVE_SOFTMAX_ONEDNN_IMPL(Algorithm::kSoftmax, dnnl::softmax_backward,
                                  dnnl::softmax_forward);
CPU_PRIMITIVE_SOFTMAX_ONEDNN_IMPL(Algorithm::kLogSoftmax, dnnl::logsoftmax_backward,
                                  dnnl::logsoftmax_forward);
#undef CPU_PRIMITIVE_SOFTMAX_ONEDNN_IMPL

template<typename SoftmaxBackwardBase, Algorithm algorithm, dnnl::memory::data_type data_type>
std::unique_ptr<SoftmaxBackwardBase> NewOneDnnSoftmaxBackward() {
  return std::unique_ptr<SoftmaxBackwardBase>(
      new OneDnnSoftmaxBackwardImpl<SoftmaxBackwardBase, algorithm, data_type>());
}

#endif  // WITH_ONEDNN

template<typename SoftmaxBackwardBase, Algorithm algorithm, typename T>
std::unique_ptr<SoftmaxBackwardBase> NewSoftmaxBackward() {
  return std::unique_ptr<SoftmaxBackwardBase>(
      new SoftmaxBackwardImpl<SoftmaxBackwardBase, algorithm, T>());
}

template<typename BackwardFactoryBase, typename SoftmaxBackwardBase, Algorithm algorithm>
class GenericSoftmaxBackwardFactoryImpl : public BackwardFactoryBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GenericSoftmaxBackwardFactoryImpl);
  GenericSoftmaxBackwardFactoryImpl() = default;
  ~GenericSoftmaxBackwardFactoryImpl() override = default;

  std::unique_ptr<SoftmaxBackwardBase> New(DataType data_type) override {
#define MAKE_NEW_SOFTMAX_BACKWARD_ENTRY(type_cpp, type_proto) \
  {type_proto, NewSoftmaxBackward<SoftmaxBackwardBase, algorithm, type_cpp>},
    static const std::map<DataType, std::function<std::unique_ptr<SoftmaxBackwardBase>()>>
        new_softmax_backward_handle{
            OF_PP_FOR_EACH_TUPLE(MAKE_NEW_SOFTMAX_BACKWARD_ENTRY, CPU_PRIMITIVE_FLOATING_TYPE_SEQ)};
#undef MAKE_NEW_SOFTMAX_BACKWARD_ENTRY

#ifdef WITH_ONEDNN
    if (OneDnnIsEnabled() && data_type == DataType::kFloat) {
      static std::function<std::unique_ptr<SoftmaxBackwardBase>()> onednn_f32_softmax_backward =
          NewOneDnnSoftmaxBackward<SoftmaxBackwardBase, algorithm, dnnl::memory::data_type::f32>;
      return onednn_f32_softmax_backward();
    }
#endif
    return NewPrimitiveFromHandlers(new_softmax_backward_handle, data_type);
  }
};

using SoftmaxBackwardFactoryImpl =
    GenericSoftmaxBackwardFactoryImpl<SoftmaxBackwardFactory, SoftmaxBackward, Algorithm::kSoftmax>;
using LogSoftmaxBackwardFactoryImpl =
    GenericSoftmaxBackwardFactoryImpl<LogSoftmaxBackwardFactory, LogSoftmaxBackward,
                                      Algorithm::kLogSoftmax>;
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, SoftmaxBackwardFactory, SoftmaxBackwardFactoryImpl);
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, LogSoftmaxBackwardFactory,
                           LogSoftmaxBackwardFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
