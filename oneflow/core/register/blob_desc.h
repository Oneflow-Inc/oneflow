#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/register/pod_desc.h"
#include "oneflow/core/job/job_desc.h"
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

  static const int32_t kAlignSize = 256;

  BlobDesc& operator=(const BlobDesc&);

  void SetShape(const Shape&);
  void SetLoD(int64_t num_of_lod_levels);
  void SetOpaqueHeader(const StructPodDesc& header_pod_desc, int64_t header_byte_size);

  const Shape& shape() const { return body_.shape(); }
  DataType data_type() const { return body_.data_type(); }
  void set_data_type(DataType val) { body_.set_data_type(val); }

  int64_t num_of_lod_levels() const { return num_of_lod_levels_; }
  bool is_body_disabled() const { return is_body_disabled_; }
  void set_is_body_disabled(bool val) { is_body_disabled_ = val; }
  bool header_is_opaque() const { return header_is_opaque_; }
  bool is_dynamic() const { return is_dynamic_; }
  void set_is_dynamic(bool);

  bool operator==(const BlobDesc&) const;
  void ToProto(BlobDescProto*) const;

  void CopyFrom(const BlobDesc&);
  // legacy interface, shouldn't use in new code
  void CopyMetaFrom(const BlobDesc& other) { CopyFrom(other); }
  void CopyAllFrom(const BlobDesc& other) { CopyMetaFrom(other); }

 private:
  void InitFromProto(const BlobDescProto& proto);

  TensorPodDesc body_;
  StructPodDesc header_;
  int64_t num_of_lod_levels_;  // lod: level of details
  bool is_body_disabled_;
  bool is_dynamic_;

  // TODO(niuchong): remove opaque_header
  TensorPodDesc opaque_header_;
  bool header_is_opaque_;
};

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc);

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
