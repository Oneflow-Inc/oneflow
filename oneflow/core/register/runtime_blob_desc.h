#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

class RtBlobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtBlobDesc);
  RtBlobDesc() = delete;
  ~RtBlobDesc() = default;

  RtBlobDesc(const BlobDesc&);
  RtBlobDesc(const BlobDescProto&);

  const StructPodDesc& header_pod_desc() const { return header_; }
  const BlobDescProto& blob_desc_proto() const { return blob_desc_proto_; }
  bool is_body_disabled() const { return blob_desc_proto().is_body_disabled(); }
  int64_t num_of_lod_levels() const { return blob_desc_proto().num_of_lod_levels(); }

  DataType data_type() const { return body_.data_type(); }
  int64_t NumAxes() const { return body_.shape().NumAxes(); }
  int64_t Capacity() const { return body_.shape().elem_cnt() * GetSizeOfDataType(data_type()); }
  const Shape& body_shape() const { return body_.shape(); }

  size_t RealByteSizeOfBlobHeader() const;
  size_t RealByteSizeOfBlobBody() const;
  size_t AlignedByteSizeOfBlobBody(size_t align_size) const;
  size_t AlignedTotalByteSize(size_t align_size) const;

  bool operator==(const RtBlobDesc& rhs) const;

 private:
  void InitFromProto(const BlobDescProto& proto);

  BlobDescProto blob_desc_proto_;
  TensorPodDesc body_;
  StructPodDesc header_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
