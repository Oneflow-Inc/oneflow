#ifndef ONEFLOW_CORE_REGISTER_FIELD_DESC_H_
#define ONEFLOW_CORE_REGISTER_FIELD_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/field_desc.pb.h"

namespace oneflow {

class FieldDesc {
 public:
  ~FieldDesc() = default;

  FieldDesc();
  FieldDesc(const FieldDesc& other);
  FieldDesc(const Shape& shape, DataType data_type);
  FieldDesc(const Shape& shape) : FieldDesc() { shape_ = shape; }
  FieldDesc(const FieldDescProto& proto);

  void InitFromProto(const FieldDescProto& proto);

  const Shape& shape() const { return shape_; }
  Shape& mut_shape() { return shape_; }

  DataType data_type() const { return data_type_; }
  void set_data_type(DataType val) { data_type_ = val; }

  void ToProto(FieldDescProto* proto) const;
  bool operator==(const FieldDesc& rhs) const;

  size_t ByteSize() const;
  size_t AlignedByteSize() const;

  std::string DebugStr() const { return shape_.DebugStr() + "," + std::to_string(data_type_); }

 private:
  Shape shape_;
  DataType data_type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_FIELD_DESC_H_
