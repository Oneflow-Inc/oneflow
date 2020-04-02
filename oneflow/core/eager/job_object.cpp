#include "oneflow/core/eager/job_object.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace eager {

void JobObject::InitLogicalObjectId2OpConf() {
  for (const auto& op_conf : job_->net().op()) {
    CHECK(op_conf.has_logical_object_id());
    CHECK(logical_object_id2op_conf_.emplace(op_conf.logical_object_id(), &op_conf).second);
  }
}

void JobObject::InitLbi2LogicalObjectId() {
  for (const auto& pair : job_->helper().lbn2logical_object_id()) {
    CHECK(lbi2logical_object_id_.emplace(GenLogicalBlobId(pair.first), pair.second).second);
  }
}

bool JobObject::HasOpConf(int64_t op_logical_object_id) const {
  return logical_object_id2op_conf_.find(op_logical_object_id) != logical_object_id2op_conf_.end();
}

const OperatorConf& JobObject::OpConf4LogicalObjectId(int64_t op_logical_object_id) const {
  const auto& iter = logical_object_id2op_conf_.find(op_logical_object_id);
  CHECK(iter != logical_object_id2op_conf_.end());
  return *iter->second;
}

int64_t JobObject::LogicalObjectId4Lbi(const LogicalBlobId& lbi) const {
  return lbi2logical_object_id_.at(lbi);
}

}  // namespace eager
}  // namespace oneflow
