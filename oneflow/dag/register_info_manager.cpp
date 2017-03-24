#include "dag/register_info_manager.h"
#include <glog/logging.h>

namespace oneflow {

void RegisterInfoManager::AddProducedRegisterInfoForNonBoxingTask(
  const RegisterInfo& register_info) {
  int64_t group_id = register_info.group_id();
  CHECK(produced_group_id_to_register_info_.count(group_id) == 0);
  produced_group_id_to_register_info_.insert({ group_id, register_info });
  RegisterType type = register_info.register_type();
  CHECK(register_type_to_produced_group_id_.count(type) == 0);
  register_type_to_produced_group_id_.insert({ type, group_id });
}

void RegisterInfoManager::RemoveProducedRegisterInfoForNonBoxingTask(
  RegisterType type, int64_t group_id) {
  CHECK(produced_group_id_to_register_info_.count(group_id));
  produced_group_id_to_register_info_.erase(group_id);
  CHECK(register_type_to_produced_group_id_.count(type));
  register_type_to_produced_group_id_.erase(type);
}

void RegisterInfoManager::AddProducedRegisterInfoForBoxingTask(
  const RegisterInfo& register_info, const std::string& consumer_segment) {
  int64_t group_id = register_info.group_id();
  CHECK(produced_group_id_to_register_info_.count(group_id) == 0);
  produced_group_id_to_register_info_.insert({ group_id, register_info });
  auto group_ids_it
    = consumer_segment_to_produced_group_ids_.find(consumer_segment);
  if (group_ids_it == consumer_segment_to_produced_group_ids_.end()) {
    std::vector<int64_t> group_ids{ group_id };
    consumer_segment_to_produced_group_ids_.insert(
      { consumer_segment, group_ids });
  } else {
    group_ids_it->second.push_back(group_id);
  }
}

int64_t RegisterInfoManager::GetProducedGroupIdForNonBoxingTask(
  RegisterType type) const {
  auto group_id_it = register_type_to_produced_group_id_.find(type);
  CHECK(group_id_it != register_type_to_produced_group_id_.end());
  return group_id_it->second;
}

int64_t RegisterInfoManager::GetProducedGroupIdForBoxingTask(
  const std::string& consumer_segment, int32_t order) const {
  auto group_ids_it
    = consumer_segment_to_produced_group_ids_.find(consumer_segment);
  CHECK(group_ids_it != consumer_segment_to_produced_group_ids_.end());
  CHECK(group_ids_it->second.size() > order);
  return group_ids_it->second[order];
}

const RegisterInfo& RegisterInfoManager::GetProducedRegisterInfo(
  int64_t group_id) const {
  auto register_info_it = produced_group_id_to_register_info_.find(group_id);
  CHECK(register_info_it != produced_group_id_to_register_info_.end());
  return register_info_it->second;
}

void RegisterInfoManager::SetProducedGroupSize(int64_t group_id,
  int32_t group_size) {
  CHECK_EQ(produced_group_id_to_group_size_.count(group_id), 0);
  produced_group_id_to_group_size_.insert({ group_id, group_size });
}

int32_t RegisterInfoManager::GetProducedGroupSize(int64_t group_id) const {
  auto group_size_it = produced_group_id_to_group_size_.find(group_id);
  CHECK(group_size_it != produced_group_id_to_group_size_.end());
  return group_size_it->second;
}

void RegisterInfoManager::AddConsumerOfGroupId(
  int32_t consumer_id, int64_t group_id) {
  CHECK(produced_group_id_to_register_info_.count(group_id) > 0);
  auto group_ids_it = consumer_id_to_group_ids_.find(consumer_id);
  if (group_ids_it == consumer_id_to_group_ids_.end()) {
    consumer_id_to_group_ids_.insert({ consumer_id, { group_id } });
  } else {
    group_ids_it->second.push_back(group_id);
  }

  auto consumers_it = group_id_to_consumer_ids_.find(group_id);
  if (consumers_it == group_id_to_consumer_ids_.end()) {
    group_id_to_consumer_ids_.insert({ group_id, { consumer_id } });
  } else {
    consumers_it->second.push_back(consumer_id);
  }
}

std::vector<int64_t> RegisterInfoManager::GetProducedGroupIds() const {
  std::vector<int64_t> produced_group_ids;
  for (auto& produced_pair : produced_group_id_to_register_info_) {
    produced_group_ids.push_back(produced_pair.first);
  }
  return produced_group_ids;
}

std::vector<int64_t> RegisterInfoManager::GetGroupIdsConsumedByOthers() const {
  std::vector<int64_t> group_ids_consumed_by_others;
  for (auto& group_id_consumer_pair : group_id_to_consumer_ids_) {
    group_ids_consumed_by_others.push_back(group_id_consumer_pair.first);
  }
  return group_ids_consumed_by_others;
}

std::vector<int32_t> RegisterInfoManager::GetConsumersOfGroupId(
  int64_t group_id) const {
  auto consumers_it = group_id_to_consumer_ids_.find(group_id);
  CHECK(consumers_it != group_id_to_consumer_ids_.end());
  return consumers_it->second;
}

void RegisterInfoManager::AddConsumedGroupId(int64_t group_id) {
  // CHECK(consumed_register_info_group_ids_.count(group_id) == 0);
  consumed_register_info_group_ids_.insert(group_id);
}

std::vector<int64_t> RegisterInfoManager::GetConsumedGroupIds() const {
  std::vector<int64_t> consumed_group_ids;
  for (auto& group_id : consumed_register_info_group_ids_) {
    consumed_group_ids.push_back(group_id);
  }
  return consumed_group_ids;
}

RegisterInfo RegisterInfoManager::CompleteProducedRegisterInfoCrossPath(
  RegisterType produced_register_type,
  const RegisterInfo& consumed_register_info) {
  auto group_id_it
    = register_type_to_produced_group_id_.find(produced_register_type);
  CHECK(group_id_it != register_type_to_produced_group_id_.end());
  RegisterInfo& produced_register_info
    = produced_group_id_to_register_info_[group_id_it->second];

  auto& blob_names_inside_envelope
    = consumed_register_info.GetBlobNamesInsideEnvelope();
  for (auto& blob_name : blob_names_inside_envelope) {
    produced_register_info.AddEmptyBlob(blob_name, EnvelopeFlag::kInEnvelope);
  }

  auto& blob_names_outside_envelope
    = consumed_register_info.GetBlobNamesOutsideEnvelope();
  for (auto& blob_name : blob_names_outside_envelope) {
    produced_register_info.AddEmptyBlob(blob_name, EnvelopeFlag::kOutEnvelope);
  }

  auto& envelope_blobs = consumed_register_info.GetEnvelopeNames();
  for (auto& blob_name : envelope_blobs) {
    produced_register_info.AddEmptyBlob(blob_name, EnvelopeFlag::kOnEnvelope);
  }

  return produced_register_info;
}

}  // namespace oneflow