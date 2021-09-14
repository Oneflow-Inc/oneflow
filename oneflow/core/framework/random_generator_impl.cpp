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
#include "oneflow/core/framework/random_generator_impl.h"

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/functional/functional.h"
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace one {

struct CPUGeneratorState {
  static constexpr int64_t state_size = std::mt19937::state_size;  // 624
  int64_t states[state_size];
  int64_t seed;
};
constexpr int64_t CPUGeneratorState::state_size;

Maybe<Tensor> CPUGeneratorImpl::GetState() const {
  CPUGeneratorState state;
  const auto& device = JUST(Device::New("cpu"));
  const auto& tensor_state = JUST(functional::Empty(Shape{sizeof(state)}, DType::Char(), device));

  std::stringstream ss;
  ss << engine_;
  auto splits = Split(ss.str(), " ");
  // The last element in `splits` indicates state size, not state.
  if (splits.size() != CPUGeneratorState::state_size + 1) {
    return Error::RuntimeError() << "std::mt19937 state size should be "
                                 << CPUGeneratorState::state_size << ", but got "
                                 << splits.size() - 1;
  }
  for (int i = 0; i < CPUGeneratorState::state_size; ++i) {
    state.states[i] = std::atoll(splits.at(i).c_str());
  }
  state.seed = current_seed();

  const auto& callback = std::make_shared<std::function<void(uint64_t)>>([&](uint64_t of_blob_ptr) {
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    memcpy(of_blob->mut_blob()->mut_dptr<char>(), &state, sizeof(state));
  });
  JUST(SpinCounter::SpinWait(1, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
    return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(JUST(tensor_state->AsMirroredTensor()), sc, callback,
                                               "mut");
    });
  }));
  return tensor_state;
}

Maybe<void> CPUGeneratorImpl::SetState(const std::shared_ptr<Tensor>& tensor_state) {
  CPUGeneratorState state;
  if (tensor_state->shape()->elem_cnt() != sizeof(state)) {
    return Error::RuntimeError() << "Tensor state size is not match for CPU generator. It needs "
                                 << sizeof(state) << ", but got "
                                 << tensor_state->shape()->elem_cnt();
  }
  const auto& callback = std::make_shared<std::function<void(uint64_t)>>([&](uint64_t of_blob_ptr) {
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    memcpy(&state, of_blob->blob().dptr<char>(), sizeof(state));
  });
  JUST(SpinCounter::SpinWait(1, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
    return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(JUST(tensor_state->AsMirroredTensor()), sc, callback,
                                               "const");
    });
  }));

  set_current_seed(state.seed);
  std::stringstream ss;
  for (int i = 0; i < CPUGeneratorState::state_size; ++i) { ss << state.states[i] << " "; }
  ss << CPUGeneratorState::state_size;
  ss >> engine_;
  return Maybe<void>::Ok();
}

#ifdef WITH_CUDA
namespace {

int GetThreadNum(const cudaDeviceProp& prop) {
  switch (prop.major) {
    case 3:  // Kepler
      return 2 * 192;
    case 5:  // Maxwell
      return 2 * 128;
    case 6:  // Pascal
      if ((prop.minor == 1) || (prop.minor == 2)) {
        return 2 * 128;
      } else {
        return 2 * 64;
      }
    case 7:  // Volta and Turing
      return 2 * 64;
    default: return 2 * 64;
  }
}

}  // namespace

CUDAGeneratorImpl::CUDAGeneratorImpl(uint64_t seed, int device_index)
    : DeviceGeneratorImpl(seed, detail::DeviceKey{DeviceType::kGPU, device_index}) {
  cudaDeviceProp prop;
  OF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  max_block_num_ = prop.multiProcessorCount;
  max_thread_num_ = GetThreadNum(prop);
  OF_CUDA_CHECK(
      cudaMalloc(&curand_states_, max_block_num_ * max_thread_num_ * sizeof(curandState)));
  detail::InitCurandStates(seed, max_block_num_, max_thread_num_, curand_states_);
}

CUDAGeneratorImpl::~CUDAGeneratorImpl() { OF_CUDA_CHECK(cudaFree(curand_states_)); }

void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  detail::InitCurandStates(seed_, max_block_num_, max_thread_num_, curand_states_);
}

Maybe<Tensor> CUDAGeneratorImpl::GetState() const {
  int64_t state_size = max_block_num_ * max_thread_num_ * sizeof(curandState);
  int64_t total_size = state_size + sizeof(int64_t);
  const auto& device = JUST(Device::New("cpu"));
  const auto& tensor_state = JUST(functional::Empty(Shape{total_size}, DType::Char(), device));

  const auto& callback = std::make_shared<std::function<void(uint64_t)>>([&](uint64_t of_blob_ptr) {
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    OF_CUDA_CHECK(cudaMemcpy(of_blob->mut_blob()->mut_dptr<char>(), &curand_states_, state_size,
                             cudaMemcpyDefault));
    memcpy(of_blob->mut_blob()->mut_dptr<char>() + state_size, &seed_, sizeof(int64_t));
  });
  JUST(SpinCounter::SpinWait(1, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
    return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(JUST(tensor_state->AsMirroredTensor()), sc, callback,
                                               "mut");
    });
  }));
  return tensor_state;
}

Maybe<void> CUDAGeneratorImpl::SetState(const std::shared_ptr<Tensor>& tensor_state) {
  int64_t state_size = max_block_num_ * max_thread_num_ * sizeof(curandState);
  int64_t total_size = state_size + sizeof(int64_t);
  if (tensor_state->shape()->elem_cnt() != total_size) {
    return Error::RuntimeError() << "Tensor state size is not match for CUDA generator. It needs "
                                 << total_size << ", but got " << tensor_state->shape()->elem_cnt();
  }
  const auto& callback = std::make_shared<std::function<void(uint64_t)>>([&](uint64_t of_blob_ptr) {
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    const char* data = of_blob->blob().dptr<char>();
    uint64_t seed = *((uint64_t*)(data + state_size));
    set_current_seed(seed);
    OF_CUDA_CHECK(
        cudaMemcpy(&curand_states_, of_blob->blob().dptr<char>(), state_size, cudaMemcpyDefault));
  });
  JUST(SpinCounter::SpinWait(1, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
    return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(JUST(tensor_state->AsMirroredTensor()), sc, callback,
                                               "const");
    });
  }));
  return Maybe<void>::Ok();
}
#endif  // WITH_CUDA

Maybe<Tensor> AutoGeneratorImpl::GetState() const { OF_UNIMPLEMENTED(); }

Maybe<void> AutoGeneratorImpl::SetState(const std::shared_ptr<Tensor>& tensor_state) {
  OF_UNIMPLEMENTED();
}

namespace detail {

bool operator==(const DeviceKey& k1, const DeviceKey& k2) {
  return k1.device_type == k2.device_type && k1.device_index == k2.device_index;
}

size_t DeviceKeyHash::operator()(const DeviceKey& key) const {
  return (static_cast<uint64_t>(key.device_type) << 10) + key.device_index;
}

template<>
DeviceKey MakeDeviceKey<CPUGeneratorImpl>(int device_index) {
  return DeviceKey{DeviceType::kCPU, 0};
}

template<>
Maybe<CPUGeneratorImpl> MakeGeneratorImpl<CPUGeneratorImpl>(uint64_t seed, int device_index) {
  return std::make_shared<CPUGeneratorImpl>(seed);
}

#ifdef WITH_CUDA
int GetCudaDeviceCount() {
  /* static */ int cuda_device_count;
  OF_CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
  return cuda_device_count;
}

template<>
DeviceKey MakeDeviceKey<CUDAGeneratorImpl>(int device_index) {
  if (device_index == -1) { OF_CUDA_CHECK(cudaGetDevice(&device_index)); }
  return DeviceKey{DeviceType::kGPU, device_index};
}

template<>
Maybe<CUDAGeneratorImpl> MakeGeneratorImpl<CUDAGeneratorImpl>(uint64_t seed, int device_index) {
  CHECK_OR_RETURN(device_index >= 0 && device_index < GetCudaDeviceCount())
      << "Invalid device index " << device_index;
  return std::make_shared<CUDAGeneratorImpl>(seed, device_index);
}
#endif  // WITH_CUDA

}  // namespace detail

}  // namespace one
}  // namespace oneflow
