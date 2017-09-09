#include "oneflow/core/schedule/utilization_util.h"
namespace oneflow {
namespace schedule {

#define EVENT_TYPE_FIELD_TUPLE_SEQ                                \
  OF_PP_MAKE_TUPLE_SEQ(kTaskStreamResource, task_stream_resource) \
  OF_PP_MAKE_TUPLE_SEQ(kRegstResource, regst_resource)

void UtilizationUtil::SetResourceType(const UtilizationEventProto& event_proto,
                                      UtilizationProto* utilization_proto) {
  switch (event_proto.resource_type_case()) {
#define FIELD_COPY_CASE_ENTRY(resouce_type_case, resource_type_field) \
  case UtilizationEventProto::resouce_type_case:                      \
    *utilization_proto->mutable_##resource_type_field() =             \
        event_proto.resource_type_field();                            \
    break;

    OF_PP_FOR_EACH_TUPLE(FIELD_COPY_CASE_ENTRY, EVENT_TYPE_FIELD_TUPLE_SEQ);
    default: UNEXPECTED_RUN();
  }
}

std::string UtilizationUtil::CreateUniqueName(
    const UtilizationEventProto& event_proto) {
  switch (event_proto.resource_type_case()) {
#define RESOURCE_TYPE_CASE_ENTRY(resouce_type_case, resource_type_field) \
  case UtilizationEventProto::resouce_type_case:                         \
    return UtilizationUtil::CreateUniqueName(event_proto.resource_type_field());

    OF_PP_FOR_EACH_TUPLE(RESOURCE_TYPE_CASE_ENTRY, EVENT_TYPE_FIELD_TUPLE_SEQ);
    default: UNEXPECTED_RUN();
  }
  return "";
}

std::string UtilizationUtil::CreateUniqueName(
    const TaskStreamResource& task_stream_res) {
  return "TaskStreamResource," + std::to_string(task_stream_res.task_id()) + ","
         + std::to_string(task_stream_res.stream_id());
}

std::string UtilizationUtil::CreateUniqueName(const RegstResource& regst_res) {
  return "RegstResource," + std::to_string(regst_res.regst_desc_id()) + ","
         + std::to_string(regst_res.regst_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kComputation>(
    const UtilizationResource& resource) {
  CHECK(resource.has_computation());
  return std::string(OF_PP_STRINGIZE(kComputation));
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kDevComputation>(
    const UtilizationResource& resource) {
  CHECK(resource.has_dev_computation());
  return std::string(OF_PP_STRINGIZE(kDevComputation)) + "-"
         + std::to_string(resource.dev_computation().device_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kStream>(
    const UtilizationResource& resource) {
  CHECK(resource.has_stream());
  return std::string(OF_PP_STRINGIZE(kStream)) + "-"
         + std::to_string(resource.stream().device_id()) + "-"
         + std::to_string(resource.stream().stream_id());
}

template<>
std::string UtilizationUtil::GetResourceUniqueName<UtilizationResource::kTask>(
    const UtilizationResource& resource) {
  CHECK(resource.has_task());
  return std::string(OF_PP_STRINGIZE(kTask)) + "-"
         + std::to_string(resource.task().task_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kTaskStream>(
    const UtilizationResource& resource) {
  CHECK(resource.has_task_stream());
  return std::string(OF_PP_STRINGIZE(kTaskStream)) + "-"
         + std::to_string(resource.task_stream().task_id()) + "-"
         + std::to_string(resource.task_stream().stream_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kMemory>(
    const UtilizationResource& resource) {
  CHECK(resource.has_memory());
  return std::string(OF_PP_STRINGIZE(kMemory));
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kDevMemory>(
    const UtilizationResource& resource) {
  CHECK(resource.has_dev_memory());
  return std::string(OF_PP_STRINGIZE(kDevMemory)) + "-"
         + std::to_string(resource.dev_memory().device_id());
}

template<>
std::string
UtilizationUtil::GetResourceUniqueName<UtilizationResource::kRegstDesc>(
    const UtilizationResource& resource) {
  CHECK(resource.has_regst_desc());
  return std::string(OF_PP_STRINGIZE(kRegstDesc)) + "-"
         + std::to_string(resource.regst_desc().regst_desc_id());
}

template<>
std::string UtilizationUtil::GetResourceUniqueName<UtilizationResource::kRegst>(
    const UtilizationResource& resource) {
  CHECK(resource.has_regst());
  return std::string(OF_PP_STRINGIZE(kRegst)) + "-"
         + std::to_string(resource.regst().regst_desc_id()) + "-"
         + std::to_string(resource.regst().regst_id());
}

std::string UtilizationUtil::GetUniqueName(
    const UtilizationResource& resource) {
  switch (resource.resource_type_case()) {
#define UTILIZATION_RESOURCE_CASE_ENTRY(type_case, class_name) \
  case type_case: return GetResourceUniqueName<type_case>(resource);
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
  auto task = ugraph.sgraph()->node_mgr().Find(resource.task().task_id());
  CHECK(task);
  UtilizationResource dc;
  dc.mutable_dev_computation()->set_device_id(task->device()->id());
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

  auto task =
      ugraph.sgraph()->node_mgr().Find(resource.task_stream().task_id());
  CHECK(task);

  UtilizationResource s;
  s.mutable_stream()->set_device_id(task->device()->id());
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
  auto regst_desc = ugraph.sgraph()->regst_desc_mgr().Find(
      resource.regst_desc().regst_desc_id());
  CHECK(regst_desc);
  dm.mutable_dev_memory()->set_device_id(
      regst_desc->owner_task()->device()->id());
  cb(dm);
}

template<>
void UtilizationUtil::ForEachGroupedResource<UtilizationResource::kRegst>(
    const UtilizationResource& resource, const UtilizationGraph& ugraph,
    const std::function<void(const UtilizationResource&)>& cb) {
  CHECK(resource.has_regst());
  UtilizationResource rd;
  rd.mutable_regst_desc()->set_regst_desc_id(
      resource.regst_desc().regst_desc_id());
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

}  // namespace oneflow
}  // namespace oneflow
