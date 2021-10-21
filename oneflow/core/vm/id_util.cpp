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
#include <climits>
#include <glog/logging.h>
#include "oneflow/core/vm/id_util.h"

namespace oneflow {
namespace vm {

namespace {

static const int64_t kObjectIdMaximumValue = LLONG_MAX / 2;
static const int64_t kMachineNumberLimit = (1 << 12);
static const int64_t kErrorCodeLimit = 4096;

static_assert(kMachineNumberLimit >= kErrorCodeLimit, "");

int64_t ObjectIdCounter() {
  static int64_t counter = 0;
  return (counter += kMachineNumberLimit);
}

int64_t NewLogicalObjectIdFromCounter() { return ObjectIdCounter() + kMachineNumberLimit - 1; }

int64_t NewPhysicalObjectIdFromCounter(int32_t machine_id) {
  CHECK_LT(machine_id, kMachineNumberLimit - 1);
  return ObjectIdCounter() + machine_id;
}

}  // namespace

int64_t IdUtil::IsErrorId(int64_t id) { return id >= -kErrorCodeLimit && id <= kErrorCodeLimit; }

int64_t IdUtil::NewLogicalValueObjectId() {
  int64_t val = NewLogicalObjectIdFromCounter();
  CHECK_LT(val, kObjectIdMaximumValue);
  return val;
}

int64_t IdUtil::NewLogicalValueSymbolId() {
  return NewLogicalObjectIdFromCounter() + kObjectIdMaximumValue;
}

int64_t IdUtil::IsLogicalValueId(int64_t id) {
  CHECK(IsValueId(id));
  return ((id + 1) % kObjectIdMaximumValue) == 0;
}

int64_t IdUtil::NewPhysicalValueObjectId(int32_t machine_id) {
  int64_t val = NewPhysicalObjectIdFromCounter(machine_id);
  CHECK_LT(val, kObjectIdMaximumValue);
  return val;
}

int64_t IdUtil::NewPhysicalValueSymbolId(int32_t machine_id) {
  return NewPhysicalObjectIdFromCounter(machine_id) + kObjectIdMaximumValue;
}

bool IdUtil::IsObjectId(int64_t object_id) { return object_id < kObjectIdMaximumValue; }

bool IdUtil::IsSymbolId(int64_t symbol_id) { return symbol_id > kObjectIdMaximumValue; }

int64_t IdUtil::GetTypeId(int64_t id) {
  if (IsTypeId(id)) { return id; }
  return -id;
}

bool IdUtil::IsTypeId(int64_t id) { return id < 0; }

int64_t IdUtil::GetValueId(int64_t id) {
  if (IsValueId(id)) { return id; }
  return -id;
}

bool IdUtil::IsValueId(int64_t id) { return id > 0; }

}  // namespace vm
}  // namespace oneflow
