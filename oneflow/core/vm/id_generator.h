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
#ifndef ONEFLOW_CORE_VM_ID_GENERATOR_H_
#define ONEFLOW_CORE_VM_ID_GENERATOR_H_

#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace vm {

class IdGenerator {
 public:
  virtual ~IdGenerator() = default;

  virtual Maybe<int64_t> NewSymbolId() = 0;
  virtual Maybe<int64_t> NewObjectId() = 0;

 protected:
  IdGenerator() = default;
};

class LogicalIdGenerator : public IdGenerator {
 public:
  LogicalIdGenerator(const LogicalIdGenerator&) = delete;
  LogicalIdGenerator(LogicalIdGenerator&&) = delete;
  LogicalIdGenerator() = default;
  ~LogicalIdGenerator() override = default;

  Maybe<int64_t> NewSymbolId() override;
  Maybe<int64_t> NewObjectId() override;
};

class PhysicalIdGenerator : public IdGenerator {
 public:
  PhysicalIdGenerator(const PhysicalIdGenerator&) = delete;
  PhysicalIdGenerator(PhysicalIdGenerator&&) = delete;
  PhysicalIdGenerator() = default;
  ~PhysicalIdGenerator() override = default;

  Maybe<int64_t> NewSymbolId() override;
  Maybe<int64_t> NewObjectId() override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ID_GENERATOR_H_
