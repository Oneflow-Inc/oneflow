#include "oneflow/core/register/chunk_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

ChunkDesc::ChunkDesc() : ChunkDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType()) {}

ChunkDesc::ChunkDesc(const Shape& shape, DataType data_type)
    : shape_(shape), data_type_(data_type) {}

ChunkDesc::ChunkDesc(const ChunkDescProto& proto) {
  shape_ = Shape(proto.shape());
  data_type_ = proto.data_type();
}

void ChunkDesc::ToProto(ChunkDescProto* proto) const {
  shape_.ToProto(proto->mutable_shape());
  proto->set_data_type(data_type_);
}

bool ChunkDesc::operator==(const ChunkDesc& rhs) const {
  return shape() == rhs.shape() && data_type() == rhs.data_type();
}

}  // namespace oneflow
