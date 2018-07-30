#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

FieldDesc::FieldDesc() : FieldDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType()) {}

FieldDesc::FieldDesc(const Shape& shape, DataType data_type)
    : shape_(shape), data_type_(data_type) {}

FieldDesc::FieldDesc(const FieldDescProto& proto) {
  shape_ = Shape(proto.shape());
  data_type_ = proto.data_type();
}

void FieldDesc::ToProto(FieldDescProto* proto) const {
  shape_.ToProto(proto->mutable_shape());
  proto->set_data_type(data_type_);
}

bool FieldDesc::operator==(const FieldDesc& rhs) const {
  return shape() == rhs.shape() && data_type() == rhs.data_type();
}

size_t FieldDesc::ByteSize() const { return shape_.elem_cnt() * GetSizeOfDataType(data_type_); }

size_t FieldDesc::AlignedByteSize() const { return RoundUp(ByteSize(), kCudaAlignSize); }

}  // namespace oneflow
