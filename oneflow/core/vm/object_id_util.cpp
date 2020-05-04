#include <climits>
#include <glog/logging.h>
#include "oneflow/core/vm/object_id_util.h"

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

int64_t ObjectIdUtil::IsErrorId(int64_t id) {
  return id >= -kErrorCodeLimit && id <= kErrorCodeLimit;
}

int64_t ObjectIdUtil::NewLogicalValueObjectId() {
  int64_t val = NewLogicalObjectIdFromCounter();
  CHECK_LT(val, kObjectIdMaximumValue);
  return val;
}

int64_t ObjectIdUtil::NewLogicalValueSymbolId() {
  return NewLogicalObjectIdFromCounter() + kObjectIdMaximumValue;
}

int64_t ObjectIdUtil::IsLogicalValueId(int64_t id) {
  CHECK(IsValueId(id));
  return ((id + 1) % kObjectIdMaximumValue) == 0;
}

int64_t ObjectIdUtil::NewPhysicalValueObjectId(int32_t machine_id) {
  int64_t val = NewPhysicalObjectIdFromCounter(machine_id);
  CHECK_LT(val, kObjectIdMaximumValue);
  return val;
}

int64_t ObjectIdUtil::NewPhysicalValueSymbolId(int32_t machine_id) {
  return NewPhysicalObjectIdFromCounter(machine_id) + kObjectIdMaximumValue;
}

bool ObjectIdUtil::IsObjectId(int64_t object_id) { return object_id < kObjectIdMaximumValue; }

bool ObjectIdUtil::IsSymbolId(int64_t symbol_id) { return symbol_id > kObjectIdMaximumValue; }

int64_t ObjectIdUtil::GetTypeId(int64_t id) {
  if (IsTypeId(id)) { return id; }
  return -id;
}

bool ObjectIdUtil::IsTypeId(int64_t id) { return id < 0; }

int64_t ObjectIdUtil::GetValueId(int64_t id) {
  if (IsValueId(id)) { return id; }
  return -id;
}

bool ObjectIdUtil::IsValueId(int64_t id) { return id > 0; }

}  // namespace vm
}  // namespace oneflow
