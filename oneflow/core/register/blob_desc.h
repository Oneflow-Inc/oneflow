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

  void CopyFrom(const BlobDesc&);
  void SetLoD(int64_t num_of_lod_levels);

  const Shape& shape() const { return body_.shape(); }
  Shape& mut_shape() { return *body_.mut_shape(); }

  DataType data_type() const { return body_.data_type(); }
  void set_data_type(DataType val) { body_.set_data_type(val); }

  int64_t num_of_lod_levels() const { return num_of_lod_levels_; }
  void set_num_of_lod_levels(int64_t val);
  bool is_body_disabled() const { return is_body_disabled_; }
  void set_is_body_disabled(bool val) { is_body_disabled_ = val; }

  bool operator==(const BlobDesc&) const;
  void ToProto(BlobDescProto*) const;

  // legacy interface, shouldn't use in new code
  void CopyMetaFrom(const BlobDesc& other) { CopyFrom(other); }
  void CopyAllFrom(const BlobDesc& other) { CopyMetaFrom(other); }

 private:
  void InitFromProto(const BlobDescProto& proto);

  TensorPodDesc body_;
  StructPodDesc header_;
  int64_t num_of_lod_levels_;  // lod: level of details
  bool is_body_disabled_;
};

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc);

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
