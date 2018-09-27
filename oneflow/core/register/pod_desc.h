#ifndef ONEFLOW_CORE_REGISTER_POD_DESC_H_
#define ONEFLOW_CORE_REGISTER_POD_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/pod.pb.h"

namespace oneflow {

class PodDesc {
 public:
  PodDesc() = default;
  virtual ~PodDesc() = default;

  template<typename T>
  const T* Cast() const;

  virtual size_t ByteSize() const = 0;
  virtual void ToProto(PodProto* pod_proto) const = 0;
  virtual std::unique_ptr<PodDesc> Clone() const = 0;
};

class ShapedPodDesc final : public PodDesc {
 public:
  ShapedPodDesc() = default;
  ShapedPodDesc(const Shape& shape, DataType data_type)
      : PodDesc(), shape_(shape), data_type_(data_type) {}
  explicit ShapedPodDesc(const ShapedPodProto& shape_pod);
  ~ShapedPodDesc() = default;

  const Shape& shape() const { return shape_; }
  DataType data_type() const { return data_type_; }
  Shape* mut_shape() { return &shape_; }
  void set_data_type(DataType data_type) { data_type_ = data_type; }

  size_t ByteSize() const override;
  void ToProto(PodProto* pod_proto) const override;
  std::unique_ptr<PodDesc> Clone() const override { return std::make_unique<ShapedPodDesc>(*this); }

 private:
  Shape shape_;
  DataType data_type_;
};

class AlignedFieldPodDesc;

class StructPodDesc final : public PodDesc {
 public:
  StructPodDesc() = default;
  explicit StructPodDesc(const StructPodProto&);
  explicit StructPodDesc(const StructPodDesc&);
  ~StructPodDesc() = default;

  size_t ByteSize() const override;
  void ToProto(PodProto* pod_proto) const override { ToProto(pod_proto->mutable_struct_pod()); }
  std::unique_ptr<PodDesc> Clone() const override { return std::make_unique<StructPodDesc>(*this); }
  void ToProto(StructPodProto* pod_proto) const;

  bool HasField(const std::string& name) const;
  const PodDesc& Field(const std::string& name) const;
  void AddField(const std::string& name, const Shape& shape, DataType data_type,
                size_t align_shift);
  void AddField(const std::string& name, const Shape& shape, DataType data_type) {
    AddField(name, shape, data_type, 3);
  }
  void AddCopedField(const std::string& name, const PodDesc& pod_desc, size_t align_shift);
  void AddCopedField(const std::string& name, const PodDesc& pod_desc) {
    AddCopedField(name, pod_desc, 3);
  }
  size_t PtrOffset4Field(const std::string& field_name) const;

 private:
  void InitFromProto(const StructPodProto& struct_pod);
  void AddField(std::unique_ptr<AlignedFieldPodDesc>&& field);
  void AddField(const std::string& name, std::unique_ptr<PodDesc>&& field, size_t align_shift = 3);

  std::vector<std::unique_ptr<AlignedFieldPodDesc>> fields_;
  HashMap<std::string, int32_t> name2field_idx_;
};

class AlignedFieldPodDesc final : public PodDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AlignedFieldPodDesc);
  ~AlignedFieldPodDesc() = default;

 private:
  friend class StructPodDesc;
  AlignedFieldPodDesc(const std::string& name, std::unique_ptr<PodDesc>&& field,
                      size_t align_shift = 3)
      : PodDesc(), name_(name), field_(std::move(field)), align_shift_(align_shift) {}
  explicit AlignedFieldPodDesc(const AlignedFieldPodProto& aligned_field_pod);
  size_t ByteSize() const override;
  void ToProto(PodProto* pod_proto) const override { UNIMPLEMENTED(); }
  std::unique_ptr<PodDesc> Clone() const override { UNIMPLEMENTED(); }
  void ToProto(AlignedFieldPodProto* aligned_field_proto) const;

  const PodDesc& field() const { return *field_; }
  const std::string& name() const { return name_; }

  std::string name_;
  std::unique_ptr<PodDesc> field_;
  size_t align_shift_;
};

template<typename T>
const T* PodDesc::Cast() const {
  static_assert(std::is_same<T, ShapedPodDesc>::value || std::is_same<T, StructPodDesc>::value,
                "only ShapedPodDesc and StructPodDesc supported");
  return dynamic_cast<T*>(this);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_POD_DESC_H_
