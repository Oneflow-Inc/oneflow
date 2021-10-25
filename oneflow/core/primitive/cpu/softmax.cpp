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
#include "oneflow/core/primitive/include/softmax.h"
#include "oneflow/core/primitive/include/log_softmax.h"
#include "oneflow/core/primitive/cpu/type_seq.h"

namespace oneflow {

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

  void Launch(StreamContext* stream_ctx, size_t rows, size_t cols, const void* x,
              void* y) override {
    SoftmaxCpu<algorithm, T>(rows, cols, reinterpret_cast<const T*>(x), reinterpret_cast<T*>(y));
  }
};

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
    const auto it = new_softmax_handle.find(data_type);
    if (it != new_softmax_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

using SoftmaxFactoryImpl = GenericSoftmaxFactoryImpl<SoftmaxFactory, Softmax, Algorithm::kSoftmax>;
using LogSoftmaxFactoryImpl =
    GenericSoftmaxFactoryImpl<LogSoftmaxFactory, LogSoftmax, Algorithm::kLogSoftmax>;
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, SoftmaxFactory, SoftmaxFactoryImpl);
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, LogSoftmaxFactory, LogSoftmaxFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
