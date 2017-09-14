#include "oneflow/core/schedule/utilization_graph.h"
#include "oneflow/core/schedule/bfs_visitor.h"
#include "oneflow/core/schedule/utilization_util.h"

namespace oneflow {
namespace schedule {

void UtilizationGraph::ForEachUtilizationInPath(
    Utilization* leaf, const std::function<void(Utilization*)>& cb) {
  auto foreach_next = [&](Utilization* utilization,
                          const std::function<void(Utilization*)>& cb) {
    utilization_arc_mgr().Input(utilization, cb);
  };
  auto foreach_prev = [&](Utilization* utilization,
                          const std::function<void(Utilization*)>& cb) {
    utilization_arc_mgr().Output(utilization, cb);
  };
  BfsVisitor<Utilization*> bfs_visitor(foreach_next, foreach_prev);
  bfs_visitor(leaf, cb);
}

Utilization* UtilizationGraph::FindOrCreateUtilization(
    const UtilizationResource& resource) {
  Utilization* utilization = FindUtilization(resource);
  return utilization ? utilization : CreateUtilization(resource);
}

template<typename class_name>
class_name* UtilizationGraph::FindConcreteUtilization(
    const UtilizationResource& resource) const {
  return node_mgr<class_name>().Find(UtilizationUtil::GetUniqueName(resource));
}

Utilization* UtilizationGraph::FindUtilization(
    const UtilizationResource& resource) const {
  switch (resource.resource_type_case()) {
#define FIND_UTILIZATION_ENTRY(type_case, class_name) \
  case type_case: return FindConcreteUtilization<class_name>(resource);
    OF_PP_FOR_EACH_TUPLE(FIND_UTILIZATION_ENTRY, UTILIZATION_TYPE_SEQ);
    default: UNEXPECTED_RUN();
  }
  return nullptr;
}

template<typename class_name>
class_name* UtilizationGraph::CreateConcreteUtilization(
    const UtilizationResource& resource) {
  return mut_node_mgr<class_name>()->Create(resource);
}

Utilization* UtilizationGraph::CreateUtilization(
    const UtilizationResource& resource) {
  Utilization* utilization = nullptr;
  switch (resource.resource_type_case()) {
#define CREATE_UTILIZATION_ENTRY(type_case, class_name)            \
  case type_case:                                                  \
    utilization = CreateConcreteUtilization<class_name>(resource); \
    break;
    OF_PP_FOR_EACH_TUPLE(CREATE_UTILIZATION_ENTRY, UTILIZATION_TYPE_SEQ);
    default: UNEXPECTED_RUN();
  }
  utilization->CreateAscendantIfNotFound(this);
  return utilization;
}

void UtilizationGraph::InitRoot() {
  UtilizationResource computation_resource;
  computation_resource.mutable_computation();
  computation_ =
      CreateConcreteUtilization<ComputationUtilization>(computation_resource);
  CHECK(computation_);
  UtilizationResource memory_resource;
  memory_resource.mutable_memory();
  memory_ = CreateConcreteUtilization<MemoryUtilization>(memory_resource);
  CHECK(memory_);
}

void UtilizationGraph::ForEachUtilization(
    const std::function<void(Utilization*)>& cb) const {
#define UTILIZATION_MGR_FOR_EACH_ENTRY(type_case, class_name) \
  node_mgr<class_name>().MutForEach(cb);
  OF_PP_FOR_EACH_TUPLE(UTILIZATION_MGR_FOR_EACH_ENTRY, UTILIZATION_TYPE_SEQ);
}

template<typename src_node_type, typename dst_node_type>
void UtilizationGraph::ConnectConcreteArc(Utilization* src, Utilization* dst) {
  src_node_type* src_utilization = dynamic_cast<src_node_type*>(src);
  CHECK(src_utilization);
  dst_node_type* dst_utilization = dynamic_cast<dst_node_type*>(dst);
  CHECK(dst_utilization);
  mut_arc_mgr<src_node_type, dst_node_type>()->CreateIfNotFound(
      src_utilization, dst_utilization);
}

void UtilizationGraph::Connect(Utilization* src, Utilization* dst) {
  mut_utilization_arc_mgr()->CreateIfNotFound(src, dst);
  auto GetKey = [](UtilizationResource::ResourceTypeCase src_type,
                   UtilizationResource::ResourceTypeCase dst_type) {
    return std::to_string(src_type) + "-" + std::to_string(dst_type);
  };
  static const HashMap<std::string,
                       std::function<void(Utilization*, Utilization*)>>
      entries{
#define SPECIALIZED_CONNECT_ENTRY(src_node_type, dst_node_type)            \
  {GetKey(src_node_type::resource_type_case,                               \
          dst_node_type::resource_type_case),                              \
   [&](Utilization* src, Utilization* dst) {                               \
     return UtilizationGraph::ConnectConcreteArc<src_node_type,            \
                                                 dst_node_type>(src, dst); \
   }},
          OF_PP_FOR_EACH_TUPLE(SPECIALIZED_CONNECT_ENTRY, UTILIZATION_ARC_SEQ)};
  entries.at(GetKey(src->GetResourceTypeCase(), dst->GetResourceTypeCase()))(
      src, dst);
}

}  // namespace schedule
}  // namespace oneflow
