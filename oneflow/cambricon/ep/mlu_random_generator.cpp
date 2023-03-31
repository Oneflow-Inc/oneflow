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
#include "oneflow/cambricon/ep/mlu_random_generator.h"

#include "oneflow/cambricon/common/mlu_guard.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace ep {

MLUGenerator::MLUGenerator(uint64_t seed, int device_index)
    : RandomGenerator(), seed_(seed), device_index_(device_index), need_update_state_(true) {
  MluCurrentDeviceGuard guard(device_index_);
  OF_CNNL_CHECK(cnnlRandCreateGenerator(&cnnl_rng_, CNNL_RAND_RNG_MTGP32));
  this->set_current_seed(seed);
  OF_CNNL_CHECK(cnnlRandGetMTGP32StateSize(nullptr, &state_size_));
  OF_MLU_CHECK(cnrtMalloc(&state_, state_size_));
  OF_MLU_CHECK(cnrtMemset(state_, 0, state_size_));
}

MLUGenerator::~MLUGenerator() {
  MluCurrentDeviceGuard guard(device_index_);
  OF_MLU_CHECK(cnrtSyncDevice());
  OF_CNNL_CHECK(cnnlRandDestroyGenerator(cnnl_rng_));
  OF_MLU_CHECK(cnrtFree(state_));
}

void MLUGenerator::set_current_seed(uint64_t seed) {
  MluCurrentDeviceGuard guard(device_index_);
  seed_ = seed;
  need_update_state_ = true;
  OF_CNNL_CHECK(cnnlRandSetPseudoRandomGeneratorSeed(cnnl_rng_, seed_));
}

static constexpr size_t seed_size = sizeof(uint64_t);
static constexpr size_t need_update_state_size = sizeof(bool);
static constexpr size_t total_size = seed_size + need_update_state_size;

size_t MLUGenerator::GetStateSize() const { return state_size_ + total_size; }

void MLUGenerator::GetState(size_t state_size, void* state) const {
  MluCurrentDeviceGuard guard(device_index_);
  CHECK_EQ_OR_THROW(state_size, GetStateSize())
      << "the state size of mlu generator should be equal to " << GetStateSize();
  memcpy(state, &seed_, seed_size);
  memcpy(static_cast<char*>(state) + seed_size, &need_update_state_, need_update_state_size);
  OF_MLU_CHECK(
      cnrtMemcpy(static_cast<char*>(state) + total_size, state_, state_size_, cnrtMemcpyDevToHost));
}

void MLUGenerator::SetState(size_t state_size, const void* state) {
  MluCurrentDeviceGuard guard(device_index_);
  CHECK_EQ_OR_THROW(state_size, GetStateSize())
      << "the state size of mlu generator should be equal to " << GetStateSize();
  memcpy(&seed_, state, seed_size);
  memcpy(&need_update_state_, static_cast<const char*>(state) + seed_size, need_update_state_size);
  OF_MLU_CHECK(cnrtMemcpy(state_, static_cast<char*>(const_cast<void*>(state)) + total_size,
                          state_size_, cnrtMemcpyHostToDev));
}

void MLUGenerator::update_state(cnnlHandle_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (need_update_state_) {
    MluCurrentDeviceGuard guard(device_index_);
    OF_CNNL_CHECK(cnnlRandMakeMTGP32KernelState(handle, state_, nullptr, nullptr, seed_));
    need_update_state_ = false;
  }
}

template<>
std::string GetRandomGeneratorDeviceTypeName<MLUGenerator>() {
  return "mlu";
}

}  // namespace ep
}  // namespace oneflow
