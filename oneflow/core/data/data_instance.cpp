#include "oneflow/core/data/data_instance.h"
#include "oneflow/core/data/data_transform.h"

namespace oneflow {
namespace data {

void DataInstance::InitFromProto(const DataInstanceProto& proto) {
  for (const auto& field_proto : proto.data_fields()) {
    auto data_field_ptr = CreateDataFieldFromProto(field_proto);
    CHECK(AddField(std::move(data_field_ptr)));
  }
}

template<DataSourceCase dsrc, typename... Args>
DataField* DataInstance::GetOrCreateField(Args&&... args) {
  if (fields_.find(dsrc) == fields_.end()) {
    using DataFieldT = typename DataFieldTrait<dsrc>::type;
    std::unique_ptr<DataField> data_field_ptr;
    data_field_ptr.reset(new DataFieldT(std::forward<Args>(args)...));
    data_field_ptr->SetSource(dsrc);
    AddField(std::move(data_field_ptr));
  }
  return fields_.at(dsrc).get();
}

void DataInstance::Transform(const DataTransformProto& trans_proto) {
#define MAKE_CASE(trans)                       \
  case trans: {                                \
    DoDataTransform<trans>(this, trans_proto); \
    break;                                     \
  }

  switch (trans_proto.transform_case()) {
    MAKE_CASE(DataTransformProto::kResize)
    MAKE_CASE(DataTransformProto::kTargetResize)
    MAKE_CASE(DataTransformProto::kSegmentationPolyToMask)
    default: { UNIMPLEMENTED(); }
  }
#undef MAKE_CASE
}

}  // namespace data
}  // namespace oneflow
