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
#include "oneflow/core/ep/cpu/cpu_random_generator.h"

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {
namespace ep {

struct CPUGeneratorState {
  static constexpr int64_t state_size = std::mt19937::state_size;  // 624
  int64_t states[state_size] = {};
  int64_t seed = 0;
};
constexpr int64_t CPUGeneratorState::state_size;

void CPUGenerator::set_current_seed(uint64_t seed) {
  seed_ = seed;
  engine_.seed(seed_);
  torch_engine_ = pytorch_mt19937_engine(seed);
}

size_t CPUGenerator::GetStateSize() const { return sizeof(CPUGeneratorState); }

void CPUGenerator::GetState(size_t state_size, void* state) const {
  CHECK_EQ_OR_THROW(state_size, GetStateSize())
      << "state size of cpu generator should be equal to " << GetStateSize();
  CPUGeneratorState local_state;
  std::stringstream ss;
  ss << engine_;
  std::vector<std::string> splits;
  Split(ss.str(), " ", [&](std::string&& s) { splits.emplace_back(s); });
  // The last element in `splits` indicates state size, not state.
  if (splits.size() != CPUGeneratorState::state_size + 1) {
    return THROW(RuntimeError) << "std::mt19937 state size should be "
                               << CPUGeneratorState::state_size << ", but got "
                               << splits.size() - 1;
  }
  for (int i = 0; i < CPUGeneratorState::state_size; ++i) {
    local_state.states[i] = std::atoll(splits[i].data());
  }
  local_state.seed = current_seed();
  memcpy(state, &local_state, sizeof(CPUGeneratorState));
}

void CPUGenerator::SetState(size_t state_size, const void* state) {
  CHECK_EQ_OR_THROW(state_size, GetStateSize())
      << "state size of cpu generator should be equal to " << GetStateSize();
  const CPUGeneratorState* local_state = static_cast<const CPUGeneratorState*>(state);
  seed_ = local_state->seed;
  std::stringstream ss;
  for (int i = 0; i < CPUGeneratorState::state_size; ++i) { ss << local_state->states[i] << " "; }
  ss << CPUGeneratorState::state_size;
  ss >> engine_;
}

template<>
std::string GetRandomGeneratorDeviceTypeName<CPUGenerator>() {
  return "cpu";
}

}  // namespace ep
}  // namespace oneflow
