#include "dag/actor_dag_memory_allocator.h"
#include "dag/actor_dag.h"
#include "dag/actor_meta.h"
#include "dag/task_dag.h"
#include "dag/layer_meta.h"
#include "dag/dag_iterator.h"
#include "common/shape.h"
#include "layers/base_layer.h"
#include "memory/memory_manager.h"
#include "memory/blob.h"
#include <utility>
#include <string>
#include <unordered_map>

// TODO(Chongin): Currently there're some redundancy between codes for 
// different task's memory allocation. To be refined later.

namespace caffe {
template <typename Dtype>
void ActorDagMemoryAllocator<Dtype>::Alloc(ActorDag<Dtype>* actor_dag) {
  DagIterator<ActorDag<Dtype>, false> dag_iterator(*actor_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() == NodeType::kOpNode) {
      auto actor_node
        = dynamic_cast<OpNode<ActorMeta<Dtype>>*>(current_node);
      CHECK_NOTNULL(actor_node);
      auto actor_meta = actor_node->op();
      auto actor_name = actor_node->node_name();
      AllocMemoryForActorTask(actor_meta, actor_name);
    }
  }
}

template <typename Dtype>
void ActorDagMemoryAllocator<Dtype>::Free(ActorDag<Dtype>* actor_dag) {
  DagIterator<ActorDag<Dtype>, false> dag_iterator(*actor_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) {
      /*auto actor_node
        = dynamic_cast<const OpNode<ActorMeta<Dtype>>*>(current_node);*/
      auto actor_node
        = dynamic_cast<OpNode<ActorMeta<Dtype>>*>(current_node);
      CHECK_NOTNULL(actor_node);
      auto actor_meta = actor_node->op();
      auto current_name = current_node->node_name();
      FreeMemoryForActorTask(actor_meta, current_name);
    }
  }
}

template <typename Dtype>
void ActorDagMemoryAllocator<Dtype>::AllocMemoryForActorTask(
  std::shared_ptr<ActorMeta<Dtype>>& actor_meta, 
  const std::string& actor_name) {
  auto task_type = actor_meta->task_type();
  switch (task_type) {
    case TaskType::kDataTask: {
      AllocMemoryForDataTask(actor_meta->mutable_dag(), actor_name);
      break;
    }
    case TaskType::kComputeTask: {
      AllocMemoryForComputeTask(actor_meta->mutable_dag(), actor_name);
      break;
    }
  }
}

template <typename Dtype>
void ActorDagMemoryAllocator<Dtype>::AllocMemoryForDataTask(
  std::shared_ptr<TaskDag<Dtype>>& task_dag,
  const std::string& actor_name) {
  std::unordered_map<std::string,
    std::pair<DataModelBaseParam<Dtype>*, Shape >> name_to_blob_shape;
  // Pre-compute memory needed for this task
  size_t task_memory_needed = 0;
  DagIterator<TaskDag<Dtype>, false> dag_iterator(*task_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    auto task_node
      = dynamic_cast<OpNode<LayerMeta<Dtype>>*>(current_node);
    CHECK_NOTNULL(task_node);
    auto current_name = task_node->node_name();
    auto layer_meta = task_node->op();
    auto layer = layer_meta->mutable_layer();
    auto model_param = layer->GetMutableModelParam();
    auto data_param = layer->GetMutableDataParam();
    auto model_blob_names = model_param->blob_names();
    auto data_blob_names = data_param->blob_names();

    for (auto& blob_name : model_blob_names) {
      auto shape = model_param->GetShape(blob_name);
      name_to_blob_shape.insert(std::make_pair(blob_name,
        std::make_pair(model_param, shape)));
      task_memory_needed += shape.count();
    }

    for (auto& blob_name : data_blob_names) {
      auto&& shape = data_param->GetShape(blob_name);
      name_to_blob_shape.insert(std::make_pair(blob_name,
        std::make_pair(data_param, shape)));
      task_memory_needed += shape.count();
    }
  }

  // To be use for allocating memory according to machine_id
  auto machine_id = GetMachineID(actor_name);
  // size of current dtype
  size_t size_of_dtype = sizeof(Dtype);
  // Allocate memory from  memory manager
  MemoryManager::Context ctx;
  // TODO(Chonglin): Unify MemoryManager::Context::Device and MemoryType
  // LoaderLayer should be run on CPU
  ctx.dev_type = MemoryManager::Context::kCPU;
  // Just set dev_id to 0 as it's on CPU
  ctx.dev_id = 0;
  MemoryManager::Handle handle = MemoryManager::Get()->Alloc(
    task_memory_needed * size_of_dtype,
    ctx);
  auto ptr = handle.dptr;
  name_to_mem_handle_.insert({ actor_name, handle });
  name_to_comp_task_mem_.insert(std::make_pair(actor_name, name_to_blob_shape));

  // Asign memory to every blob
  size_t offset = 0;
  for (auto& item : name_to_blob_shape) {
    auto&& model = item.second.first;
    auto shape = item.second.second;
    auto blob = new Blob<Dtype>(
      static_cast<void*>(static_cast<Dtype*>(ptr)+offset),
      shape,
      MemoryType::kHostPageableMemory);
    model->SetBlob(item.first, blob);
    offset += shape.count() * size_of_dtype;
  }
}

template <typename Dtype>
void ActorDagMemoryAllocator<Dtype>::AllocMemoryForComputeTask(
  std::shared_ptr<TaskDag<Dtype>>& task_dag,
  const std::string& actor_name) {
  std::unordered_map<std::string,
    std::pair<DataModelBaseParam<Dtype>*, Shape >> name_to_blob_shape;
  // Pre-compute memory needed for this task
  size_t task_memory_needed = 0;
  DagIterator<TaskDag<Dtype>, false> dag_iterator(*task_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    auto task_node
      = dynamic_cast<OpNode<LayerMeta<Dtype>>*>(current_node);
    CHECK_NOTNULL(task_node);
    auto current_name = task_node->node_name();
    auto layer_meta = task_node->op();
    auto layer = layer_meta->mutable_layer();
    auto model_param = layer->GetMutableModelParam();
    auto data_param = layer->GetMutableDataParam();
    auto model_blob_names = model_param->blob_names();
    auto data_blob_names = data_param->blob_names();

    for (auto& blob_name : model_blob_names) {
      auto shape = model_param->GetShape(blob_name);
      auto second = std::make_pair(model_param, shape);
      name_to_blob_shape.insert(std::make_pair(blob_name,  second));
      task_memory_needed += shape.count();
    }

    for (auto& blob_name : data_blob_names) {
      auto&& shape = data_param->GetShape(blob_name);
      name_to_blob_shape.insert(std::make_pair(blob_name,
        std::make_pair(data_param, shape)));
      task_memory_needed += shape.count();
    }
  }

  // To be use for allocating memory according to machine_id
  auto machine_id = GetMachineID(actor_name);
  // size of current dtype
  size_t size_of_dtype = sizeof(Dtype);
  // Allocate memory from  memory manager
  MemoryManager::Context ctx;
  // TODO(Chonglin): Unify MemoryManager::Context::Device and MemoryType
  ctx.dev_type = MemoryManager::Context::kGPU;
  ctx.dev_id = GetLocalDeviceID(actor_name);
  MemoryManager::Handle handle = MemoryManager::Get()->Alloc(
    task_memory_needed * size_of_dtype,
    ctx);
  auto ptr = handle.dptr;
  name_to_mem_handle_.insert({ actor_name, handle });
  name_to_comp_task_mem_.insert(std::make_pair(actor_name, name_to_blob_shape));

  // Asign memory to every blob
  size_t offset = 0;
  for (auto& item : name_to_blob_shape) {
    auto&& model = item.second.first;
    auto shape = item.second.second;
    auto blob = new Blob<Dtype>(
      static_cast<void*>(static_cast<Dtype*>(ptr) + offset),
      shape,
      MemoryType::kDeviceMemory);
    model->SetBlob(item.first, blob);
    offset += shape.count() * size_of_dtype;
  }
}

template <typename Dtype>
void ActorDagMemoryAllocator<Dtype>::FreeMemoryForActorTask(
  std::shared_ptr<ActorMeta<Dtype>>& actor_meta,
  const std::string& actor_name) {
  auto task_type = actor_meta->task_type();
  switch (task_type) {
    case TaskType::kDataTask: {
      FreeMemoryForDataTask(actor_meta->mutable_dag(), actor_name);
      break;
    }
    case TaskType::kComputeTask: {
      FreeMemoryForComputeTask(actor_meta->mutable_dag(), actor_name);
      break;
    }
  }
}

template <typename Dtype>
void ActorDagMemoryAllocator<Dtype>::FreeMemoryForDataTask(
  std::shared_ptr<TaskDag<Dtype>>& task_dag,
  const std::string& actor_name) {
  CHECK(name_to_mem_handle_.find(actor_name) != name_to_mem_handle_.end());
  auto&& handle = (name_to_mem_handle_.find(actor_name))->second;
  // Blob doesn't own the alloc memory, release directly
  MemoryManager::Get()->Free(handle);
}

template <typename Dtype>
void ActorDagMemoryAllocator<Dtype>::FreeMemoryForComputeTask(
  std::shared_ptr<TaskDag<Dtype>>& task_dag,
  const std::string& actor_name) {
  CHECK(name_to_mem_handle_.find(actor_name) != name_to_mem_handle_.end());
  auto&& handle = (name_to_mem_handle_.find(actor_name))->second;
  // Blob doesn't own the alloc memory, release directly
  MemoryManager::Get()->Free(handle);
}

INSTANTIATE_CLASS(ActorDagMemoryAllocator);
}  // namespace caffe
