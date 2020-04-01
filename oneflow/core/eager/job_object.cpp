#include "oneflow/core/eager/job_object.h"

namespace oneflow {
namespace eager {

void JobObject::InitLogicalObjectId2OpConf() {
  for (const auto& op_conf : job_->net().op()) {
    CHECK(op_conf.has_logical_object_id());
    CHECK(logical_object_id2op_conf_.emplace(op_conf.logical_object_id(), &op_conf).second);
  }
}

bool JobObject::HasOpConf(int64_t op_logical_object_id) const {
  return logical_object_id2op_conf_.find(op_logical_object_id) != logical_object_id2op_conf_.end();
}

const OperatorConf& JobObject::LookupOpConf(int64_t op_logical_object_id) const {
  const auto& iter = logical_object_id2op_conf_.find(op_logical_object_id);
  CHECK(iter != logical_object_id2op_conf_.end());
  return *iter->second;
}

}  // namespace eager
}  // namespace oneflow
