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
    const auto it = new_softmax_backward_handle.find(data_type);
    if (it != new_softmax_backward_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
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
