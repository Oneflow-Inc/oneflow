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
#include "oneflow/core/primitive/include/softmax_backward.h"
#include "oneflow/core/primitive/cpu/type_seq.h"

namespace oneflow {

namespace primitive {

namespace {

template<typename T>
void SoftmaxBackwardCpu(size_t rows, size_t cols, const T* y, const T* dy, T* dx) {
  FOR_RANGE(int64_t, i, 0, rows) {
    T row_sum = 0;
    FOR_RANGE(int64_t, j, 0, cols) {
      const int64_t offset = i * cols + j;
      row_sum += y[offset] * dy[offset];
    }
    FOR_RANGE(int64_t, j, 0, cols) {
      const int64_t offset = i * cols + j;
      dx[offset] = (dy[offset] - row_sum) * y[offset];
    }
  }
}

template<typename T>
class SoftmaxBackwardImpl : public SoftmaxBackward {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxBackwardImpl);
  SoftmaxBackwardImpl() = default;
  ~SoftmaxBackwardImpl() override = default;

  using SoftmaxBackward::Launch;
  void Launch(StreamContext* stream_ctx, size_t rows, size_t cols, const void* y, const void* dy,
              void* dx) override {
    SoftmaxBackwardCpu(rows, cols, reinterpret_cast<const T*>(y), reinterpret_cast<const T*>(dy),
                       reinterpret_cast<T*>(dx));
  }
};

template<typename T>
std::unique_ptr<SoftmaxBackward> NewSoftmaxBackward() {
  return std::unique_ptr<SoftmaxBackward>(new SoftmaxBackwardImpl<T>());
}

class SoftmaxBackwardFactoryImpl : public SoftmaxBackwardFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxBackwardFactoryImpl);
  SoftmaxBackwardFactoryImpl() = default;
  ~SoftmaxBackwardFactoryImpl() override = default;

  std::unique_ptr<SoftmaxBackward> New(DataType data_type) override {
#define MAKE_NEW_SOFTMAX_ENTRY(type_cpp, type_proto) {type_proto, NewSoftmaxBackward<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<SoftmaxBackward>()>>
        new_softmax_backward_handle{
            OF_PP_FOR_EACH_TUPLE(MAKE_NEW_SOFTMAX_ENTRY, CPU_PRIMITIVE_FLOATING_TYPE_SEQ)};
#undef MAKE_NEW_SOFTMAX_ENTRY
    const auto it = new_softmax_backward_handle.find(data_type);
    if (it != new_softmax_backward_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, SoftmaxBackwardFactory, SoftmaxBackwardFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
