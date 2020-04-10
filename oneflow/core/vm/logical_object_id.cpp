#include <climits>
#include <glog/logging.h>
#include "oneflow/core/vm/logical_object_id.h"

namespace oneflow {
namespace vm {

namespace {

static const int64_t kNaiveLogicalObjectIdMaximumValue = LLONG_MAX / 2;

int64_t LogicalObjectIdCounter() {
  static int64_t counter = 1;
  return counter++;
}

}  // namespace

int64_t NewNaiveLogicalObjectId() {
  int64_t val = LogicalObjectIdCounter();
  CHECK_LT(val, kNaiveLogicalObjectIdMaximumValue);
  return val;
}
int64_t NewConstHostLogicalObjectId() {
  return LogicalObjectIdCounter() + kNaiveLogicalObjectIdMaximumValue;
}
bool IsNaiveLogicalObjectId(int64_t logical_object_id) {
  return logical_object_id < kNaiveLogicalObjectIdMaximumValue;
}
bool IsConstHostLogicalObjectId(int64_t logical_object_id) {
  return logical_object_id > kNaiveLogicalObjectIdMaximumValue;
}

int64_t GetTypeLogicalObjectId(int64_t value_logical_object_id) { return -value_logical_object_id; }
bool IsTypeLogicalObjectId(int64_t logical_object_id) { return logical_object_id < 0; }
bool IsValueLogicalObjectId(int64_t logical_object_id) { return logical_object_id > 0; }
int64_t GetSelfLogicalObjectId(int64_t logical_object_id) { return logical_object_id; }

}  // namespace vm
}  // namespace oneflow
