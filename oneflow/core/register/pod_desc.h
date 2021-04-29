/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_REGISTER_POD_DESC_H_
#define ONEFLOW_CORE_REGISTER_POD_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/pod.pb.h"

namespace std {

template<>
struct hash<oneflow::FieldId> {
  size_t operator()(const oneflow::FieldId& field_id) const {
    if (field_id.has_key()) { return std::hash<int>()(field_id.key()); }
    if (field_id.has_lbi()) { return std::hash<oneflow::LogicalBlobId>()(field_id.lbi()); }
    UNIMPLEMENTED();
  }
};

}  // namespace std

namespace oneflow {

FieldId NewFieldId(FieldKey key);
FieldId NewFieldId(const LogicalBlobId& lbi);
inline bool operator==(const FieldId& lhs, const FieldId& rhs) {
  PbMd message_diff;
  return message_diff.Equivalent(lhs, rhs);
}

class PodDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PodDesc);
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

class TensorPodDesc final : public PodDesc {
 public:
  TensorPodDesc();
  TensorPodDesc(const Shape& shape, DataType data_type);
  TensorPodDesc(const std::shared_ptr<Shape>& shape, DataType data_type);
  explicit TensorPodDesc(const TensorPodProto& shape_pod_proto);
  explicit TensorPodDesc(const TensorPodDesc& shape_pod);
  ~TensorPodDesc() = default;
  const Shape& shape() const { return *CHECK_NOTNULL(shape_.get()); }
  DataType data_type() const { return data_type_; }
  DataType* mut_data_type() { return &data_type_; }
  Shape* mut_shape() { return CHECK_NOTNULL(shape_.get()); }
  void set_data_type(DataType data_type) { data_type_ = data_type; }

  void InitFromProto(const TensorPodProto& shape_pod);

  size_t ByteSize() const override;
  void ToProto(PodProto* pod_proto) const override;
  void ToProto(TensorPodProto* pod_proto) const;
  std::unique_ptr<PodDesc> Clone() const override { return std::make_unique<TensorPodDesc>(*this); }
  bool operator==(const PodDesc& rhs) const override;

 private:
  std::shared_ptr<Shape> shape_;
  DataType data_type_;
};

class FieldPodDesc;

class StructPodDesc final : public PodDesc {
 public:
  StructPodDesc() = default;
  explicit StructPodDesc(const StructPodProto&);
  explicit StructPodDesc(const StructPodDesc&);
  ~StructPodDesc() = default;

  StructPodDesc* MutStructField(const FieldId& field_id);
  StructPodDesc* MutStructField(const FieldId& field_id, int32_t default_alignment);
  const PodDesc& Field(FieldKey field_key) const { return Field(NewFieldId(field_key)); }
  const PodDesc& Field(const FieldId& field_id) const;
  void AddField(FieldKey field_key, const PodDesc& pod_desc);
  void AddField(const FieldId& field_id, const PodDesc& pod_desc);
  void AddField(const FieldId& field_id, const PodDesc& pod_desc, size_t alignment);
  bool HasField(FieldKey field_key) const { return HasField(NewFieldId(field_key)); }
  bool HasField(const FieldId& field_id) const;
  PodDesc* MutExistedField(FieldKey field_key) { return MutExistedField(NewFieldId(field_key)); }

  std::unique_ptr<PodDesc> Clone() const override { return std::make_unique<StructPodDesc>(*this); }
  void InitFromProto(const StructPodProto& struct_pod);
  void ToProto(PodProto* pod_proto) const override { ToProto(pod_proto->mutable_struct_pod()); }
  void ToProto(StructPodProto* pod_proto) const;

  size_t ByteOffset4Field(const FieldId& field_name) const;
  size_t ByteSize() const override;

  StructPodDesc& operator=(const StructPodDesc&);
  bool operator==(const PodDesc& rhs) const override;

 private:
  PodDesc* MutExistedField(const FieldId& field_id);
  void Clear();
  void AddField(std::unique_ptr<FieldPodDesc>&& field);
  void AddField(const FieldId& field_id, std::unique_ptr<PodDesc>&& field);
  void AddField(const FieldId& field_id, std::unique_ptr<PodDesc>&& field, size_t alignment);

  std::vector<std::unique_ptr<FieldPodDesc>> fields_;
  HashMap<FieldId, int32_t> field_id2field_idx_;
};

class FieldPodDesc final : public PodDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FieldPodDesc);
  ~FieldPodDesc() = default;

 private:
  friend class StructPodDesc;
  FieldPodDesc(const FieldId& field_id, std::unique_ptr<PodDesc>&& pod, size_t alignment)
      : PodDesc(), field_id_(field_id), pod_(std::move(pod)), alignment_(alignment) {}
  explicit FieldPodDesc(const FieldPodProto& field_pod_proto);

  size_t ByteSize() const override;
  void ToProto(PodProto* pod_proto) const override { UNIMPLEMENTED(); }
  std::unique_ptr<PodDesc> Clone() const override { UNIMPLEMENTED(); }
  void ToProto(FieldPodProto* field_proto) const;
  bool operator==(const PodDesc& rhs) const override;

  const PodDesc& pod() const { return *pod_; }
  const FieldId& field_id() const { return field_id_; }
  PodDesc* mut_pod() { return pod_.get(); }

  FieldId field_id_;
  std::unique_ptr<PodDesc> pod_;
  size_t alignment_;
};

template<typename T>
const T& PodDesc::Cast() const {
  static_assert(std::is_same<T, TensorPodDesc>::value || std::is_same<T, StructPodDesc>::value,
                "only TensorPodDesc and StructPodDesc supported");
  return *dynamic_cast<const T*>(this);
}

template<typename T>
T* PodDesc::MutCast() {
  static_assert(std::is_same<T, TensorPodDesc>::value || std::is_same<T, StructPodDesc>::value,
                "only TensorPodDesc and StructPodDesc supported");
  return dynamic_cast<T*>(this);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_POD_DESC_H_
