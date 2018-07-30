#ifndef ONEFLOW_CORE_REGISTER_CHUNK_DESC_H_
#define ONEFLOW_CORE_REGISTER_CHUNK_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/chunk_desc.pb.h"

namespace oneflow {

class ChunkDesc {
 public:
  ~ChunkDesc() = default;

  ChunkDesc();
  ChunkDesc(const Shape& shape, DataType data_type);
  ChunkDesc(const Shape& shape) : ChunkDesc() { shape_ = shape; }
  ChunkDesc(const ChunkDescProto& proto);

  const Shape& shape() const { return shape_; }
  Shape& mut_shape() { return shape_; }

  DataType data_type() const { return data_type_; }
  void set_data_type(DataType val) { data_type_ = val; }

  void ToProto(ChunkDescProto* proto) const;
  bool operator==(const ChunkDesc& rhs) const;

  std::string DebugStr() const { return shape_.DebugStr() + "," + std::to_string(data_type_); }

 private:
  Shape shape_;
  DataType data_type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_CHUNK_DESC_H_
