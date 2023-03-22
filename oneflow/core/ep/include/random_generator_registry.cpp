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
#include "oneflow/core/ep/include/random_generator_registry.h"

#include <map>
#include "oneflow/core/common/throw.h"

namespace oneflow {
namespace ep {

using Creator = RandomGeneratorRegistry::Creator;

std::map<std::string, Creator>* GetRandomGenerators() {
  static std::map<std::string, Creator> generator_ctors;
  return &generator_ctors;
}

/*static*/ Creator RandomGeneratorRegistry::Lookup(const std::string& device) {
  auto* generator_ctors = GetRandomGenerators();
  auto it = generator_ctors->find(device);
  if (it == generator_ctors->end()) {
    THROW(RuntimeError) << "can not create generator for " << device
                        << " since the generator ctor has not been registered.";
  }
  return it->second;
}

/*static*/ void RandomGeneratorRegistry::Register(const std::string& device, Creator creator) {
  auto* generator_ctors = GetRandomGenerators();
  if (generator_ctors->count(device)) {
    THROW(RuntimeError) << "random generator for " << device
                        << " has been registered more than once.";
  }
  generator_ctors->emplace(device, creator);
}

}  // namespace ep
}  // namespace oneflow
