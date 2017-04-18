#ifndef _TASK_PARAM_CREATOR_H_
#define _TASK_PARAM_CREATOR_H_

#include <stdint.h>
#include <string>
#include <vector>
#include "task/task.h"
#include "task/lru_cache.h"

namespace oneflow {

template <typename Dtype>
class BaseDataModelParam;

template <typename Dtype>
class BaseLayer;

template <typename Dtype>
class Task;

template <typename Dtype>
class TaskParam;

template <typename Dtype>
class TaskParamCreator {
public:
  TaskParamCreator(int32_t cache_capacity, Task<Dtype>* task);

  std::shared_ptr<TaskParam<Dtype>> GetTaskParam(
    const std::vector<int64_t>& register_ids);

  int32_t task_id() const { return task_->task_id(); }
  bool is_net_receiver() const;
  const std::vector<std::shared_ptr<BaseLayer<Dtype>>>& ordered_layers() const;
  const std::vector<std::string>& layer_blobs_in_execution() const;
  int32_t index_of_layer_blob(const std::string& layer_blob) const;
  std::string register_blob_from_layer_blob(const std::string& layer_blob) const;

private:
  using TaskParamCache = LRUCache<std::string, std::shared_ptr<TaskParam<Dtype>>>;
private:
  Task<Dtype>* task_;
  std::unique_ptr<TaskParamCache> task_param_cache_;

  // The layers in |operators_| are sorted according to topological order
  std::vector<std::shared_ptr<BaseLayer<Dtype>>> ordered_layers_;
  std::vector<std::string> layer_blobs_in_execution_;
  std::unordered_map<std::string, int32_t> layer_blob_to_index_;
  std::unordered_map<std::string, std::string> layer_blob_to_register_blob_;

  void Init();

  std::string CreateCacheKey(const std::vector<int64_t>& register_ids) const;

  std::shared_ptr<TaskParam<Dtype>> NewTaskParam(
    const std::vector<int64_t>& register_ids);

  void UpdateTaskParam(std::shared_ptr<TaskParam<Dtype>> task_param,
    const std::vector<int64_t>& register_ids);
};
}  // namespace oneflow
#endif  // _TASK_PARAM_CREATOR_H_
