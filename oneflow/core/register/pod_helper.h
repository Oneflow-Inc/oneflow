#ifndef ONEFLOW_CORE_REGISTER_POD_HELPER_H_
#define ONEFLOW_CORE_REGISTER_POD_HELPER_H_

#include "oneflow/core/register/pod.pb.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

class PodHelper final {
 public:
  explicit PodHelper(PodProto* mut_pod_proto)
      : mut_pod_proto_(mut_pod_proto), pod_proto_(nullptr) {}
  explicit PodHelper(const PodProto& pod_proto) : mut_pod_proto_(nullptr), pod_proto_(&pod_proto) {}

  const PodProto& pod_proto() const;
  Shape GetShape() const;
  DataType GetDataType() const;

  size_t ByteSize() const;
  bool HasField(const std::string& field_name) const;
  PodHelper Field(const std::string&) const;
  size_t PtrOffset4Field(const std::string& field_name) const;

  PodHelper MutField(const std::string& field_name);
  void SetShapeAndDataType(const Shape& shape, DataType data_type);

 private:
  int32_t GetFieldOffset(const std::string& field_name) const;
  size_t PtrOffset4Field(int32_t field_idx) const;
  PodProto* mut_pod_proto();

  PodProto* mut_pod_proto_;
  const PodProto* pod_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_POD_HELPER_H_
