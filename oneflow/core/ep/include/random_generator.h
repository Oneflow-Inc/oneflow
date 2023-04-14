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
#ifndef ONEFLOW_CORE_EP_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_EP_RANDOM_GENERATOR_H_

#include <string>

namespace oneflow {
namespace ep {

class RandomGenerator {
 public:
  RandomGenerator() = default;
  virtual ~RandomGenerator() = default;

  virtual uint64_t current_seed() const = 0;
  virtual void set_current_seed(uint64_t seed) = 0;

  virtual std::string device_type_name() const = 0;
  virtual int64_t device_index() const = 0;

  virtual size_t GetStateSize() const = 0;
  virtual void GetState(size_t state_size, void* state) const = 0;
  virtual void SetState(size_t state_size, const void* state) = 0;
};

template<typename T>
std::string GetRandomGeneratorDeviceTypeName();

}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_RANDOM_GENERATOR_H_
