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
#ifndef ONEFLOW_CORE_EP_RANDOM_GENERATOR_REGISTRY_H_
#define ONEFLOW_CORE_EP_RANDOM_GENERATOR_REGISTRY_H_

#include <string>

#include "oneflow/core/common/util.h"

namespace oneflow {
namespace ep {

class Generator;

class RandomGeneratorRegistry {
 public:
  using Creator = std::function<std::shared_ptr<Generator>(uint64_t, int)>;

  static Creator Lookup(const std::string& device);

  static void Register(const std::string& device, Creator creator);
};

}  // namespace ep
}  // namespace oneflow

#define REGISTER_RANDOM_GENERATOR(device, T)                                         \
  COMMAND(::oneflow::ep::RandomGeneratorRegistry::Register(                          \
      device, [](uint64_t seed, int device_index) {                                  \
        return std::shared_ptr<::oneflow::ep::Generator>(new T(seed, device_index)); \
      }))

#endif  // ONEFLOW_CORE_EP_RANDOM_GENERATOR_REGISTRY_H_
