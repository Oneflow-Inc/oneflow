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
#include "oneflow/core/primitive/include/log_softmax_backward.h"
#include "oneflow/core/primitive/cpu/type_seq.h"

namespace oneflow {

namespace primitive {

namespace {

template<typename T>
void LogSoftmaxBackwardCpu(size_t rows, size_t cols, const T* y, const T* dy, T* dx) {
  for (size_t i = 0; i < rows; ++i) {
    size_t row_offset = i * cols;
    const T* row_y = y + row_offset;
    const T* row_dy = dy + row_offset;
    T* row_dx = dx + row_offset;
    T row_sum = 0;
    for (size_t j = 0; j < cols; ++j) { row_sum += row_dy[j]; }
    for (size_t j = 0; j < cols; ++j) { row_dx[j] = row_dy[j] - std::exp(row_y[j]) * row_sum; }
  }
}

template<typename T>
class LogSoftmaxBackwardImpl : public LogSoftmaxBackward {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogSoftmaxBackwardImpl);
  LogSoftmaxBackwardImpl() = default;
  ~LogSoftmaxBackwardImpl() override = default;

  void Launch(StreamContext* stream_ctx, size_t rows, size_t cols, const void* y, const void* dy,
              void* dx) override {
    LogSoftmaxBackwardCpu(rows, cols, reinterpret_cast<const T*>(y), reinterpret_cast<const T*>(dy),
                          reinterpret_cast<T*>(dx));
  }
};

template<typename T>
std::unique_ptr<LogSoftmaxBackward> NewLogSoftmaxBackward() {
  return std::unique_ptr<LogSoftmaxBackward>(new LogSoftmaxBackwardImpl<T>());
}

class LogSoftmaxBackwardFactoryImpl : public LogSoftmaxBackwardFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogSoftmaxBackwardFactoryImpl);
  LogSoftmaxBackwardFactoryImpl() = default;
  ~LogSoftmaxBackwardFactoryImpl() override = default;

  std::unique_ptr<LogSoftmaxBackward> New(DataType data_type) override {
#define MAKE_NEW_LOG_SOFTMAX_BACKWARD_ENTRY(type_cpp, type_proto) \
  {type_proto, NewLogSoftmaxBackward<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<LogSoftmaxBackward>()>>
        new_log_softmax_backward_handle{OF_PP_FOR_EACH_TUPLE(MAKE_NEW_LOG_SOFTMAX_BACKWARD_ENTRY,
                                                             CPU_PRIMITIVE_FLOATING_TYPE_SEQ)};
#undef MAKE_NEW_LOG_SOFTMAX_BACKWARD_ENTRY
    const auto it = new_log_softmax_backward_handle.find(data_type);
    if (it != new_log_softmax_backward_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, LogSoftmaxBackwardFactory,
                           LogSoftmaxBackwardFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
