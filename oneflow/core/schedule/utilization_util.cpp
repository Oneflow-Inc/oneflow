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
}
}  // namespace oneflow
