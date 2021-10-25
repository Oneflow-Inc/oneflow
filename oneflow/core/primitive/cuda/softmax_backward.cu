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
#include "oneflow/core/primitive/include/log_softmax_backward.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {

namespace primitive {

namespace {

enum class Algorithm {
  kSoftmax,
  kLogSoftmax,
};

template<typename T, Algorithm algorithm>
void SoftmaxBackwardGpu(cudaStream_t cuda_stream, size_t rows, size_t cols, const T* y, const T* dy,
                        T* dx) {
  using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
  cuda::softmax::DirectLoad<T, ComputeType> load_y(y, cols);
  cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy, cols);
  cuda::softmax::DirectStore<ComputeType, T> store(dx, cols);
  if (algorithm == Algorithm::kSoftmax) {
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
        cuda_stream, load_y, load_dy, store, rows, cols)));
  } else if (algorithm == Algorithm::kLogSoftmax) {
    OF_CUDA_CHECK((cuda::softmax::DispatchLogSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                         decltype(store), ComputeType>(
        cuda_stream, load_y, load_dy, store, rows, cols)));
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class SoftmaxBackwardImpl : public SoftmaxBackward {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxBackwardImpl);
  SoftmaxBackwardImpl() = default;
  ~SoftmaxBackwardImpl() override = default;

  void Launch(StreamContext* stream_ctx, size_t rows, size_t cols, const void* y, const void* dy,
              void* dx) override {
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    SoftmaxBackwardGpu<T, Algorithm::kSoftmax>(
        cuda_stream, rows, cols, reinterpret_cast<const T*>(y), reinterpret_cast<const T*>(dy),
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
            OF_PP_FOR_EACH_TUPLE(MAKE_NEW_SOFTMAX_ENTRY, CUDA_PRIMITIVE_FLOATING_TYPE_SEQ)};

#undef MAKE_NEW_SOFTMAX_ENTRY

    const auto it = new_softmax_backward_handle.find(data_type);
    if (it != new_softmax_backward_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, SoftmaxBackwardFactory, SoftmaxBackwardFactoryImpl);

template<typename T>
class LogSoftmaxBackwardImpl : public LogSoftmaxBackward {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogSoftmaxBackwardImpl);
  LogSoftmaxBackwardImpl() = default;
  ~LogSoftmaxBackwardImpl() override = default;

  void Launch(StreamContext* stream_ctx, size_t rows, size_t cols, const void* y, const void* dy,
              void* dx) override {
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    SoftmaxBackwardGpu<T, Algorithm::kLogSoftmax>(
        cuda_stream, rows, cols, reinterpret_cast<const T*>(y), reinterpret_cast<const T*>(dy),
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
#define MAKE_NEW_LOG_SOFTMAX_ENTRY(type_cpp, type_proto) \
  {type_proto, NewLogSoftmaxBackward<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<LogSoftmaxBackward>()>>
        new_log_softmax_backward_handle{
            OF_PP_FOR_EACH_TUPLE(MAKE_NEW_LOG_SOFTMAX_ENTRY, CUDA_PRIMITIVE_FLOATING_TYPE_SEQ)};

#undef MAKE_NEW_LOG_SOFTMAX_ENTRY

    const auto it = new_log_softmax_backward_handle.find(data_type);
    if (it != new_log_softmax_backward_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, LogSoftmaxBackwardFactory,
                           LogSoftmaxBackwardFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
