#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include "context/one.h"
#include "dag/node_meta.h"
#include "dag/task_dag.h"
#include "dag/dag_iterator.h"
#include "task/task_param_creator.h"
#include "layers/base_layer.h"
#include "task/task_op.h"
#include "task/device_register.h"
#include "task/task_param.h"
#include "task/node_manager.h"
#include "common/hash.h"

namespace caffe {
template <typename Dtype>
TaskParamCreator<Dtype>::TaskParamCreator(int32_t cache_capacity,
  Task<Dtype>* task) : task_param_cache_(new TaskParamCache(cache_capacity)),
  task_(task) {
  Init();
}

template <typename Dtype>
bool TaskParamCreator<Dtype>::is_net_receiver() const {
  return task_->is_net_receiver();
}

template <typename Dtype>
std::shared_ptr<TaskParam<Dtype>> TaskParamCreator<Dtype>::GetTaskParam(
  const std::vector<int64_t>& register_ids) {
  std::shared_ptr<TaskParam<Dtype>> task_param;
  std::string hash_key = CreateCacheKey(register_ids);
  if (!task_param_cache_->Get(hash_key, task_param)) {
    if (task_param_cache_->IsFull()) {
      // Reuse the task_param in cache
      task_param = task_param_cache_->RemoveLast();
      UpdateTaskParam(task_param, register_ids);
      task_param_cache_->Add(hash_key, task_param);
    } else {
      task_param = NewTaskParam(register_ids);
      task_param_cache_->Add(hash_key, task_param);
    }
  }
  return task_param;
}

template <typename Dtype>
std::string TaskParamCreator<Dtype>::CreateCacheKey(
  const std::vector<int64_t>& register_ids) const {
  std::string str_key;
  for (auto register_id : register_ids) {
    str_key += std::to_string(register_id);
  }
  return str_key;
}

template <typename Dtype>
std::shared_ptr<TaskParam<Dtype>> TaskParamCreator<Dtype>::NewTaskParam(
  const std::vector<int64_t>& register_ids) {
  std::shared_ptr<TaskParam<Dtype>> task_param(new TaskParam<Dtype>(this));
  UpdateTaskParam(task_param, register_ids);
  return task_param;
}

template <typename Dtype>
void TaskParamCreator<Dtype>::UpdateTaskParam(
  std::shared_ptr<TaskParam<Dtype>> task_param,
  const std::vector<int64_t>& register_ids) {
  for (auto& register_id : register_ids) {
    auto device_register
      = caffe::TheOne<Dtype>::node_manager()->GetDeviceRegister(register_id);
    auto blob_index = device_register->get_blob_index(this);
    auto& index_blob_pairs = blob_index.get_index();
    for (auto& index_blob_pair : index_blob_pairs) {
      task_param->set_blob(index_blob_pair.first, index_blob_pair.second);
    }
  }
}

template <typename Dtype>
void TaskParamCreator<Dtype>::Init() {
  ordered_layers_ = task_->task_dag()->GetOrderedLayers();
  layer_blobs_in_execution_ = task_->task_dag()->GetLayerBlobsInExecution();
  int32_t blob_idx = 0;
  for (auto& layer_blob : layer_blobs_in_execution_) {
    layer_blob_to_index_.insert({ layer_blob, blob_idx });
    ++blob_idx;
  }

  // Init |layer_blob_to_register_blob_|
  for (auto& layer_blob : layer_blobs_in_execution_) {
    auto register_blob
      = task_->task_dag()->register_blob_from_layer_blob(layer_blob);
    layer_blob_to_register_blob_.insert({ layer_blob, register_blob });
  }
}

template <typename Dtype>
const std::vector<std::shared_ptr<BaseLayer<Dtype>>>&
TaskParamCreator<Dtype>::ordered_layers() const {
  return ordered_layers_;
}

template <typename Dtype>
const std::vector<std::string>& TaskParamCreator<Dtype>::layer_blobs_in_execution(
  ) const {
  return layer_blobs_in_execution_;
}

template <typename Dtype>
int32_t TaskParamCreator<Dtype>::index_of_layer_blob(
  const std::string& layer_blob) const {
  auto layer_blob_to_index_it = layer_blob_to_index_.find(layer_blob);
  CHECK(layer_blob_to_index_it != layer_blob_to_index_.end());
  return layer_blob_to_index_it->second;
}

template <typename Dtype>
std::string TaskParamCreator<Dtype>::register_blob_from_layer_blob(
  const std::string& layer_blob) const {
  auto register_blob_it = layer_blob_to_register_blob_.find(layer_blob);
  CHECK(register_blob_it != layer_blob_to_register_blob_.end());
  return register_blob_it->second;
}

INSTANTIATE_CLASS(TaskParamCreator);
}  // namespace caffe
