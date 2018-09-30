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
  const T& Cast() const;
  template<typename T>
  T* MutCast();

  virtual size_t ByteSize() const = 0;
  virtual void ToProto(PodProto* pod_proto) const = 0;
  virtual std::unique_ptr<PodDesc> Clone() const = 0;
  virtual bool operator==(const PodDesc& rhs) const = 0;
  bool operator!=(const PodDesc& rhs) const { return !(*this == rhs); }
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
  bool operator==(const PodDesc& rhs) const override;

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

  StructPodDesc* MutStructField(const std::string& name);
  const PodDesc& Field(const std::string& name) const;
  const ShapedPodDesc& ShapedField(const std::string& name) const;
  const StructPodDesc& StructField(const std::string& name) const;
  void AddField(const std::string& name, const PodDesc& pod_desc);
  size_t ByteSize() const override;
  void InitFromProto(const StructPodProto& struct_pod);

  bool HasField(const std::string& name) const;
  StructPodDesc& operator=(const StructPodDesc&);
  std::unique_ptr<PodDesc> Clone() const override { return std::make_unique<StructPodDesc>(*this); }
  void ToProto(PodProto* pod_proto) const override { ToProto(pod_proto->mutable_struct_pod()); }
  void ToProto(StructPodProto* pod_proto) const;
  StructPodDesc* MutStructField(const std::string& name, int32_t default_align_size);
  void AddField(const std::string& name, const PodDesc& pod_desc, size_t align_size);
  bool operator==(const PodDesc& rhs) const override;
  size_t PtrOffset4Field(const std::string& field_name) const;

 private:
  void Clear();
  PodDesc* MutExistedField(const std::string& name);
  void AddField(std::unique_ptr<AlignedFieldPodDesc>&& field);
  void AddField(const std::string& name, std::unique_ptr<PodDesc>&& field);
  void AddField(const std::string& name, std::unique_ptr<PodDesc>&& field, size_t align_size);

  std::vector<std::unique_ptr<AlignedFieldPodDesc>> fields_;
  HashMap<std::string, int32_t> name2field_idx_;
};

class AlignedFieldPodDesc final : public PodDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AlignedFieldPodDesc);
  ~AlignedFieldPodDesc() = default;

 private:
  friend class StructPodDesc;
  AlignedFieldPodDesc(const std::string& name, std::unique_ptr<PodDesc>&& field, size_t align_size)
      : PodDesc(), name_(name), field_(std::move(field)), align_size_(align_size) {}
  explicit AlignedFieldPodDesc(const AlignedFieldPodProto& aligned_field_pod);

  size_t ByteSize() const override;
  void ToProto(PodProto* pod_proto) const override { UNIMPLEMENTED(); }
  std::unique_ptr<PodDesc> Clone() const override { UNIMPLEMENTED(); }
  void ToProto(AlignedFieldPodProto* aligned_field_proto) const;
  bool operator==(const PodDesc& rhs) const override;

  const PodDesc& field() const { return *field_; }
  const std::string& name() const { return name_; }
  PodDesc* mut_field() { return field_.get(); }

  std::string name_;
  std::unique_ptr<PodDesc> field_;
  size_t align_size_;
};

template<typename T>
const T& PodDesc::Cast() const {
  static_assert(std::is_same<T, ShapedPodDesc>::value || std::is_same<T, StructPodDesc>::value,
                "only ShapedPodDesc and StructPodDesc supported");
  return *dynamic_cast<const T*>(this);
}

template<typename T>
T* PodDesc::MutCast() {
  static_assert(std::is_same<T, ShapedPodDesc>::value || std::is_same<T, StructPodDesc>::value,
                "only ShapedPodDesc and StructPodDesc supported");
  return dynamic_cast<T*>(this);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_POD_DESC_H_
