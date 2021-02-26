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
#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/register/pod_desc.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

class BlobDesc final {
 public:
  BlobDesc() = delete;
  ~BlobDesc() = default;
  BlobDesc(const Shape&, DataType);
  explicit BlobDesc(DataType dtype) : BlobDesc(Shape(), dtype) {}
  explicit BlobDesc(const BlobDescProto& proto);
  explicit BlobDesc(const BlobDesc&);
  BlobDesc(const TensorPodDesc& body, bool is_tensor_list, bool is_dynamic)
      : body_(body), is_tensor_list_(is_tensor_list), is_dynamic_(is_dynamic) {}

  static const int32_t kAlignSize = 512;

  BlobDesc& operator=(const BlobDesc&);

  void set_is_tensor_list(bool val) { is_tensor_list_ = val; }
  void SetOpaqueHeader(const StructPodDesc& header_pod_desc);

  const Shape& shape() const { return body_.shape(); }
  Shape& mut_shape() { return *body_.mut_shape(); }

  DataType data_type() const { return body_.data_type(); }
  void set_data_type(DataType val) { body_.set_data_type(val); }

  bool is_tensor_list() const { return is_tensor_list_; }
  bool header_is_opaque() const { return opaque_header_ != nullptr; }
  bool is_dynamic() const { return is_dynamic_; }
  void set_is_dynamic(bool);

  bool operator==(const BlobDesc&) const;
  void ToProto(BlobDescProto*) const;

  void CopyFrom(const BlobDesc&);

 private:
  void InitFromProto(const BlobDescProto& proto);

  TensorPodDesc body_;
  bool is_tensor_list_;
  bool is_dynamic_;

  // TODO(niuchong): remove opaque_header
  std::unique_ptr<StructPodDesc> opaque_header_;
};

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc);

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
