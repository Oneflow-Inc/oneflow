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
#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

class RtBlobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtBlobDesc);
  RtBlobDesc() = delete;
  ~RtBlobDesc() = default;

  explicit RtBlobDesc(const BlobDesc& blob_desc);
  explicit RtBlobDesc(const BlobDescProto& blob_desc_proto);

  const StructPodDesc& header_pod_desc() const { return header_; }
  bool is_tensor_list() const { return is_tensor_list_; }
  bool is_dynamic() const { return is_dynamic_; }
  bool header_is_opaque() const { return header_is_opaque_; }

  DataType data_type() const { return body_.data_type(); }
  int64_t NumAxes() const { return body_.shape().NumAxes(); }
  int64_t Capacity() const { return body_.shape().elem_cnt() * GetSizeOfDataType(data_type()); }
  const Shape& body_shape() const { return body_.shape(); }
  const TensorPodDesc& body() const { return body_; }

  size_t ByteSizeOfBlobHeader() const;
  size_t ByteSizeOfBlobBody() const;
  size_t AlignedByteSizeOfBlobBody() const;
  size_t AlignedTotalByteSize() const;

  bool operator==(const RtBlobDesc& rhs) const;

 private:
  void InitFromProto(const BlobDescProto& proto);

  TensorPodDesc body_;
  StructPodDesc header_;
  bool is_tensor_list_;
  bool is_dynamic_;
  bool header_is_opaque_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
