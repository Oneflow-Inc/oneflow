#include "dag/blob_info_manager.h"
#include <glog/logging.h>

namespace oneflow {
void BlobInfoManager::RegisterBlob(const std::string& layer_blob,
  const std::string& task_blob, const std::string& logical_blob, bool is_input) {
  auto task_blob_it = layer_blob_to_task_blob_.find(layer_blob);
  // Ensure each |layer_blob| is added just once
  CHECK(task_blob_it == layer_blob_to_task_blob_.end());
  layer_blob_to_task_blob_.insert({ layer_blob, task_blob });
  layer_blobs_.push_back(layer_blob);
  if (is_input) {
    input_task_blobs_.insert(task_blob);
  }
  task_blobs_.insert(task_blob);
  auto logical_blob_it = task_blob_to_logical_blob_.find(task_blob);
  if (logical_blob_it != task_blob_to_logical_blob_.end()) return;
  task_blob_to_logical_blob_.insert({ task_blob, logical_blob });
  return;
}

void BlobInfoManager::AddProducedBlobToRegister(
  const std::string& layer_blob, int64_t group_id) {
  layer_blobs_in_execution_.insert(layer_blob);

  auto task_blob = task_blob_from_layer_blob(layer_blob);
  CHECK(task_blobs_.count(task_blob));

  // For a produced blob, the |register_blob| is the same as the |task_blob|
  auto register_blob = task_blob;
  layer_blob_to_register_blob_.insert({ layer_blob, register_blob });

  if (produced_task_blobs_.count(task_blob) == 0) {
    produced_task_blobs_.insert(task_blob);
    task_blob_to_group_id_.insert({ task_blob, group_id });
    produced_task_blob_to_group_id_.insert({ task_blob, group_id });
    task_blob_to_register_blob_.insert({ task_blob, register_blob });
    AddLogicalBlobTaskBlobMap(task_blob, group_id);
  }
}

void BlobInfoManager::RemoveProducedBlobFromRegister(
  const std::string& layer_blob, int64_t group_id) {
  CHECK(layer_blobs_in_execution_.count(layer_blob));
  layer_blobs_in_execution_.erase(layer_blob);

  auto task_blob = task_blob_from_layer_blob(layer_blob);
  CHECK(task_blobs_.count(task_blob));

  CHECK(layer_blob_to_register_blob_.count(layer_blob));
  layer_blob_to_register_blob_.erase(layer_blob);

  CHECK(produced_task_blobs_.count(task_blob));
  produced_task_blobs_.erase(task_blob);
  task_blob_to_group_id_.erase(task_blob);
  produced_task_blob_to_group_id_.erase(task_blob);
  task_blob_to_register_blob_.erase(task_blob);
  RemoveLogicalBlobTaskBlobMap(group_id);
}

void BlobInfoManager::AddConsumedBlobToRegister(const std::string& layer_blob,
  const std::string& register_blob, int64_t group_id) {
  layer_blobs_in_execution_.insert(layer_blob);

  auto task_blob = task_blob_from_layer_blob(layer_blob);
  CHECK(task_blobs_.count(task_blob));

  layer_blob_to_register_blob_.insert({ layer_blob, register_blob });
  if (consumed_task_blobs_.count(task_blob) == 0) {
    consumed_task_blobs_.insert(task_blob);
    task_blob_to_group_id_.insert({ task_blob, group_id });
    consumed_task_blob_to_group_id_.insert({ task_blob, group_id });
    task_blob_to_register_blob_.insert({ task_blob, register_blob });
  }
}

void BlobInfoManager::AddLogicalBlobTaskBlobMap(
  const std::string& task_blob, int64_t group_id) {
  // Get the |logical_blob| name corresponding to |task_blob|, which must have a
  // corresponding |logical_blob|.
  auto logical_blob_it = task_blob_to_logical_blob_.find(task_blob);
  CHECK(logical_blob_it != task_blob_to_logical_blob_.end());
  std::string logical_blob = logical_blob_it->second;

  auto logical_task_it = produced_group_id_to_logical_task_map_.find(group_id);
  if (logical_task_it == produced_group_id_to_logical_task_map_.end()) {
    // It is the first time to insert |group_id|
    LogicalBlobTaskBlobMap logical_blob_to_task_blob;
    logical_blob_to_task_blob.insert({ logical_blob, task_blob });
    produced_group_id_to_logical_task_map_.insert(
      { group_id, logical_blob_to_task_blob });
  } else {
    // The |group_id| already exists. The |logical_blob| must not be inserted
    // before, since we ensure in a particular register, each |logical_blob| has
    // a one-to-one correspondence to |task_blob|
    CHECK(logical_task_it->second.count(logical_blob) == 0);
    logical_task_it->second.insert({ logical_blob, task_blob });
  }
}

void BlobInfoManager::RemoveLogicalBlobTaskBlobMap(int64_t group_id) {
  CHECK(produced_group_id_to_logical_task_map_.count(group_id));
  produced_group_id_to_logical_task_map_.erase(group_id);
}

void BlobInfoManager::SetBlobShape(const std::string& task_blob,
  const Shape& shape) {
  if (task_blob_to_shape_.count(task_blob) > 0) {
    CHECK(task_blob_to_shape_[task_blob] == shape);
  } else {
    task_blob_to_shape_.insert({ task_blob, shape });
  }
}

Shape BlobInfoManager::GetBlobShape(const std::string& task_blob) const {
  auto shape_it = task_blob_to_shape_.find(task_blob);
  CHECK(shape_it != task_blob_to_shape_.end());
  return shape_it->second;
}

std::string BlobInfoManager::task_blob_from_logical_blob(
  int64_t group_id, const std::string& logical_blob) const {
  auto logical_task_it = produced_group_id_to_logical_task_map_.find(group_id);
  CHECK(logical_task_it != produced_group_id_to_logical_task_map_.end());
  auto task_blob_it = logical_task_it->second.find(logical_blob);
  CHECK(task_blob_it != logical_task_it->second.end());
  return task_blob_it->second;
}

void BlobInfoManager::RemoveLayerAndTaskBlobPair(const std::string& layer_blob) {
  auto task_blob_it = layer_blob_to_task_blob_.find(layer_blob);
  CHECK(task_blob_it != layer_blob_to_task_blob_.end());
  layer_blob_to_task_blob_.erase(task_blob_it);
}

void BlobInfoManager::EraseInputTaskBlob(const std::string& task_blob) {
  CHECK(input_task_blobs_.count(task_blob) > 0);
  input_task_blobs_.erase(task_blob);
}

std::vector<std::string> BlobInfoManager::input_task_blobs() const {
  std::vector<std::string> input_task_blobs;
  for (auto& input_task_blob : input_task_blobs_) {
    input_task_blobs.push_back(input_task_blob);
  }
  return input_task_blobs;
}

bool BlobInfoManager::IsProduced(const std::string& task_blob) const {
  return produced_task_blobs_.count(task_blob) > 0;
}

std::vector<std::string> BlobInfoManager::task_blobs() const {
  std::vector<std::string> all_task_blobs;
  for (auto& task_blob : task_blobs_) {
    all_task_blobs.push_back(task_blob);
  }
  return all_task_blobs;
}

std::vector<std::string> BlobInfoManager::consumed_task_blobs() const {
  std::vector<std::string> consumed_task_blobs;
  for (auto& task_blob : consumed_task_blobs_) {
    consumed_task_blobs.push_back(task_blob);
  }
  return consumed_task_blobs;
}

std::vector<std::string> BlobInfoManager::produced_task_blobs() const {
  std::vector<std::string> produced_task_blobs;
  for (auto& task_blob : produced_task_blobs_) {
    produced_task_blobs.push_back(task_blob);
  }
  return produced_task_blobs;
}

std::vector<std::string> BlobInfoManager::layer_blobs() const {
  return layer_blobs_;
}

std::vector<std::string> BlobInfoManager::layer_blobs_in_execution() const {
  std::vector<std::string> layer_blobs_in_execution;
  for (auto& layer_blob : layer_blobs_in_execution_) {
    layer_blobs_in_execution.push_back(layer_blob);
  }
  return layer_blobs_in_execution;
}

std::string BlobInfoManager::logical_blob_from_task_blob(
  const std::string& task_blob) const {
  auto logical_blob_it = task_blob_to_logical_blob_.find(task_blob);
  CHECK(logical_blob_it != task_blob_to_logical_blob_.end());
  return logical_blob_it->second;
}

std::string BlobInfoManager::task_blob_from_layer_blob(
  const std::string& layer_blob) const {
  auto task_blob_it = layer_blob_to_task_blob_.find(layer_blob);
  CHECK(task_blob_it != layer_blob_to_task_blob_.end());
  return task_blob_it->second;
}

std::string BlobInfoManager::register_blob_from_layer_blob(
  const std::string& layer_blob) const {
  auto register_blob_it = layer_blob_to_register_blob_.find(layer_blob);
  CHECK(register_blob_it != layer_blob_to_register_blob_.end());
  return register_blob_it->second;
}

std::string BlobInfoManager::register_blob_from_task_blob(
  const std::string& task_blob) const {
  auto register_blob_it = task_blob_to_register_blob_.find(task_blob);
  CHECK(register_blob_it != task_blob_to_register_blob_.end());
  return register_blob_it->second;
}

int64_t BlobInfoManager::group_id_of_task_blob(
  const std::string& task_blob) const {
  auto group_id_it = task_blob_to_group_id_.find(task_blob);
  CHECK(group_id_it != task_blob_to_group_id_.end());
  return group_id_it->second;
}

}  // namespace oneflow