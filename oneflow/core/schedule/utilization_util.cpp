#include "oneflow/core/schedule/utilization_util.h"
#include "oneflow/core/schedule/utilization.h"
#include "oneflow/core/schedule/utilization_graph.h"
namespace oneflow {
namespace schedule {

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kComputation>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_computation());
  return std::string(OF_PP_STRINGIZE(kComputation));
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kDevComputation>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_dev_computation());
  return std::string(OF_PP_STRINGIZE(kDevComputation)) + sep
         + std::to_string(resource.dev_computation().device_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kStream>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_stream());
  return std::string(OF_PP_STRINGIZE(kStream)) + sep
         + std::to_string(resource.stream().device_id()) + sep
         + std::to_string(resource.stream().stream_id());
}

template<>
std::string UtilizationUtil::GetResourceUniqueName<UtilizationResource::kTask>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_task());
  return std::string(OF_PP_STRINGIZE(kTask)) + sep
         + std::to_string(resource.task().task_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kTaskStream>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_task_stream());
  return std::string(OF_PP_STRINGIZE(kTaskStream)) + sep
         + std::to_string(resource.task_stream().task_id()) + sep
         + std::to_string(resource.task_stream().stream_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kMemory>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_memory());
  return std::string(OF_PP_STRINGIZE(kMemory));
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kDevMemory>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_dev_memory());
  return std::string(OF_PP_STRINGIZE(kDevMemory)) + sep
         + std::to_string(resource.dev_memory().device_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kRegstDesc>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_regst_desc());
  return std::string(OF_PP_STRINGIZE(kRegstDesc)) + sep
         + std::to_string(resource.regst_desc().regst_desc_id());
}

template<>
std::string UtilizationUtil::GetResourceUniqueName<UtilizationResource::kRegst>(
    const UtilizationResource& resource, const std::string& sep) {
  CHECK(resource.has_regst());
  return std::string(OF_PP_STRINGIZE(kRegst)) + sep
         + std::to_string(resource.regst().regst_desc_id()) + sep
         + std::to_string(resource.regst().regst_id());
}

std::string UtilizationUtil::GetUniqueName(const UtilizationResource& resource,
                                           const std::string& sep) {
  switch (resource.resource_type_case()) {
#define UTILIZATION_RESOURCE_CASE_ENTRY(type_case, class_name) \
  case type_case: return GetResourceUniqueName<type_case>(resource, sep);
    OF_PP_FOR_EACH_TUPLE(UTILIZATION_RESOURCE_CASE_ENTRY, UTILIZATION_TYPE_SEQ)
    default: UNEXPECTED_RUN();
  }
  return "";
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kComputation>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_computation());
}

template<>
void UtilizationUtil::ForEachGroupedResource<
    UtilizationResource::kDevComputation>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_dev_computation());
  UtilizationResource c;
  c.mutable_computation();
  cb(c);
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kStream>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_stream());
  UtilizationResource dc;
  dc.mutable_dev_computation()->set_device_id(resource.stream().device_id());
  cb(dc);
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kTask>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_task());
  auto task = ugraph.sgraph().node_mgr().Find(resource.task().task_id());
  CHECK(task);
  UtilizationResource dc;
  dc.mutable_dev_computation()->set_device_id(task->device().id());
  cb(dc);
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kTaskStream>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_task_stream());
  UtilizationResource t;
  t.mutable_task()->set_task_id(resource.task_stream().task_id());
  cb(t);

  auto task = ugraph.sgraph().node_mgr().Find(resource.task_stream().task_id());
  CHECK(task);

  UtilizationResource s;
  s.mutable_stream()->set_device_id(task->device().id());
  s.mutable_stream()->set_stream_id(resource.task_stream().stream_id());
  cb(s);
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kMemory>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_memory());
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kDevMemory>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_dev_memory());
  UtilizationResource m;
  m.mutable_memory();
  cb(m);
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kRegstDesc>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_regst_desc());
  UtilizationResource dm;
  auto regst_desc = ugraph.sgraph().regst_desc_mgr().Find(
      resource.regst_desc().regst_desc_id());
  CHECK(regst_desc);
  dm.mutable_dev_memory()->set_device_id(
      regst_desc->owner_task().device().id());
  cb(dm);
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kRegst>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_regst());
  UtilizationResource rd;
  rd.mutable_regst_desc()->set_regst_desc_id(resource.regst().regst_desc_id());
  cb(rd);
}

void UtilizationUtil::ForEachGrouped(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  switch (resource.resource_type_case()) {
#define FOR_EACH_GROUPED_RESOURCE_CASE_ENTRY(type_case, class_name) \
  case type_case:                                                   \
    return ForEachGroupedResource<type_case>(resource, ugraph, cb);
    OF_PP_FOR_EACH_TUPLE(FOR_EACH_GROUPED_RESOURCE_CASE_ENTRY,
                         UTILIZATION_TYPE_SEQ)
    default: UNEXPECTED_RUN();
  }
}

}  // namespace schedule
}  // namespace oneflow
