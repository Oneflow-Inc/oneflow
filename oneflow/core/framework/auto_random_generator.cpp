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
#include "oneflow/core/framework/auto_random_generator.h"

#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/platform/include/pthread_fork.h"

namespace oneflow {
namespace one {

struct AutoGeneratorState {
  uint64_t seed = 0;
  int64_t num = 0;
  int64_t device_tag_length = 0;
  int64_t state_length = 0;
  // std::vector<int64_t> state_sizes[num];
  // std::vector<uint8_t> device_tags[device_tag_length];
  // std::vector<uint8_t> states[state_sizes[0] + state_sizes[1] + ... + state_sizes[num - 1]]
};

void AutoGenerator::set_current_seed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(mutex_);
  seed_ = seed;
  for (const auto& it : generators_) {
    if (unlikely(pthread_fork::IsForkedSubProcess() && it.first->type() != "cpu")) { continue; }
    it.second->set_current_seed(seed);
  }
}

size_t AutoGenerator::GetStateSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t state_size = sizeof(AutoGeneratorState) + generators_.size() * sizeof(uint64_t);
  std::stringstream ss;
  auto it = generators_.begin();
  if (it != generators_.end()) {
    ss << it->second->device_type_name() << ":" << it->second->device_index();
    ++it;
  }
  for (; it != generators_.end(); ++it) {
    ss << "," << it->second->device_type_name() << ":" << it->second->device_index();
  }
  state_size += ss.str().size();
  for (const auto& it : generators_) { state_size += it.second->GetStateSize(); }
  return state_size;
}

void AutoGenerator::GetState(size_t state_size, void* state) const {
  std::lock_guard<std::mutex> lock(mutex_);
  AutoGeneratorState state_info;
  state_info.seed = current_seed();
  state_info.num = generators_.size();
  state_info.state_length = 0;
  std::vector<int64_t> state_sizes;
  state_sizes.reserve(generators_.size());

  for (auto it = generators_.begin(); it != generators_.end(); ++it) {
    state_sizes.emplace_back(it->second->GetStateSize());
    state_info.state_length += state_sizes.back();
  }
  std::stringstream ss;
  auto it = generators_.begin();
  if (it != generators_.end()) {
    ss << it->second->device_type_name() << ":" << it->second->device_index();
    ++it;
  }
  for (; it != generators_.end(); ++it) {
    ss << "," << it->second->device_type_name() << ":" << it->second->device_index();
  }

  std::string device_tags = ss.str();
  state_info.device_tag_length = device_tags.size();
  size_t total_size = sizeof(AutoGeneratorState) + state_info.num * sizeof(int64_t)
                      + state_info.device_tag_length + state_info.state_length;
  CHECK_EQ_OR_THROW(state_size, total_size)
      << "the state size of auto generator should be equal to " << total_size;
  {
    uint8_t* data = static_cast<uint8_t*>(state);
    memcpy(data, &state_info, sizeof(AutoGeneratorState));
    data += sizeof(AutoGeneratorState);
    memcpy(data, state_sizes.data(), state_info.num * sizeof(int64_t));
    data += state_info.num * sizeof(int64_t);
    memcpy(data, device_tags.data(), state_info.device_tag_length);
    data += state_info.device_tag_length;
    int i = 0;
    for (auto it = generators_.begin(); it != generators_.end(); ++it, ++i) {
      it->second->GetState(state_sizes[i], data);
      data += state_sizes[i];
    }
  }
}

void AutoGenerator::SetState(size_t state_size, const void* state) {
  AutoGeneratorState state_info;
  const uint8_t* data = static_cast<const uint8_t*>(state);
  memcpy(reinterpret_cast<void*>(&state_info), data, sizeof(AutoGeneratorState));
  if (state_size
      != sizeof(AutoGeneratorState) + state_info.num * sizeof(int64_t)
             + state_info.device_tag_length + state_info.state_length) {
    return THROW(RuntimeError) << "Invalid auto generator state, size is not match.";
  }
  data += sizeof(AutoGeneratorState);
  std::vector<int64_t> state_sizes(state_info.num);
  std::vector<const void*> state_data(state_info.num);
  memcpy(state_sizes.data(), data, state_info.num * sizeof(int64_t));
  data += state_info.num * sizeof(int64_t);
  std::string device_tags;
  device_tags.resize(state_info.device_tag_length);
  memcpy(const_cast<char*>(device_tags.data()), data, state_info.device_tag_length);
  data += state_info.device_tag_length;

  for (int i = 0; i < state_info.num; ++i) {
    state_data[i] = data;
    data += state_sizes[i];
  }
  // set current seed.
  set_current_seed(state_info.seed);

  std::vector<std::string> splits;
  Split(device_tags, ",", [&](std::string&& s) { splits.emplace_back(s); });
  if (splits.size() != state_info.num) {
    return THROW(RuntimeError) << "Invalid auto generator state. The number of state is "
                               << state_info.num << ", but device tags number is " << splits.size();
  }
  std::lock_guard<std::mutex> lock(mutex_);

  for (int i = 0; i < splits.size(); ++i) {
    const auto& device = CHECK_JUST(Device::ParseAndNew(splits[i]));
    auto generator = CHECK_JUST(GetOrCreate(device->type(), device->device_id()));
    generator->SetState(state_sizes[i], state_data[i]);
  }
}

Maybe<ep::RandomGenerator> AutoGenerator::GetOrCreate(const std::string& device, int device_index) {
  if (device_index == -1) { device_index = (device == "cpu" ? 0 : GlobalProcessCtx::LocalRank()); }
  std::lock_guard<std::mutex> lock(mutex_);
  auto device_key = JUST(Device::New(device, device_index));
  auto it = generators_.find(device_key);
  if (it == generators_.end()) {
    auto device_type = ep::DeviceManagerRegistry::GetDeviceTypeByDeviceTypeName(device);
    if (device_type == DeviceType::kInvalidDevice) {
      return Error::RuntimeError() << "Expected one of " << PrintGeneratorAvailableDevices()
                                   << " device type at start of device string: " << device;
    }
    auto device_mgr = Singleton<ep::DeviceManagerRegistry>::Get()->GetDeviceManager(device_type);
    it = generators_.emplace(device_key, device_mgr->CreateRandomGenerator(seed_, device_index))
             .first;
  }
  return it->second;
}

}  // namespace one
}  // namespace oneflow
