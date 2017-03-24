#ifndef _DAG_ACTOR_DAG_MEMORY_ALLOCATOR_H_
#define _DAG_ACTOR_DAG_MEMORY_ALLOCATOR_H_

#include <memory>
#include <unordered_map>
#include <string>
#include <functional>
#include "memory/memory_manager.h"
#include "common/str_util.h"
#include <glog/logging.h>

namespace oneflow {
template <typename Dtype>
class ActorDag;

template <typename Dtype>
class ActorMeta;

template <typename Dtype>
class TaskDag;

class Shape;

template <typename Dtype>
class DataModelBaseParam;

template <typename Dtype>
class ActorDagMemoryAllocator {
 public:
  ActorDagMemoryAllocator() = default;
  ~ActorDagMemoryAllocator() = default;

  void Alloc(ActorDag<Dtype>* actor_dag);
  void Free(ActorDag<Dtype>* actor_dag);

 private:
  void AllocMemoryForActorTask(
    std::shared_ptr<ActorMeta<Dtype>>& actor_meta,
    const std::string& actor_name);

  void AllocMemoryForDataTask(
    std::shared_ptr<TaskDag<Dtype>>& task_dag,
    const std::string& actor_name);
  void AllocMemoryForComputeTask(
    std::shared_ptr<TaskDag<Dtype>>& task_dag,
    const std::string& actor_name);

  void FreeMemoryForActorTask(
    std::shared_ptr<ActorMeta<Dtype>>& actor_meta,
    const std::string& actor_name);

  void FreeMemoryForDataTask(
    std::shared_ptr<TaskDag<Dtype>>& task_dag,
    const std::string& actor_name);
  void FreeMemoryForComputeTask(
    std::shared_ptr<TaskDag<Dtype>>& task_dag,
    const std::string& actor_name);

  std::function<int(std::string)> GetLocalDeviceID = [](
    const std::string& name) {
    auto&& fields = strings::Split(name, "_");
    CHECK_GE(fields.size(), 3) << "Actor name error!";
    return atoi(fields.back().substr(1).c_str());
  };

  std::function<int(std::string)> GetMachineID = [](
    const std::string& name) {
    auto&& fields = strings::Split(name, "_");
    CHECK_GE(fields.size(), 3) << "Actor name error!";
    return atoi(fields[fields.size() - 2].c_str());
  };

  typedef std::unordered_map<std::string, 
    std::pair<DataModelBaseParam<Dtype>*, Shape>> TaskMemMapType;

  std::unordered_map<std::string, MemoryManager::Handle> name_to_mem_handle_;
  std::unordered_map<std::string, TaskMemMapType> name_to_comp_task_mem_;
};
}  // namespace oneflow
#endif  // _DAG_ACTOR_DAG_MEMORY_ALLOCATOR_H_
