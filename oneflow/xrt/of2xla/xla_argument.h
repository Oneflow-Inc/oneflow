#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_

#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/xrt/of2xla/xla_utility.h"

namespace oneflow {
namespace mola {

class Argument {
 public:
  Argument() {}
  Argument(const LogicalBlobId &blob_id, const BlobDesc &blob_desc) {
    blob_id_.CopyFrom(blob_id);
    blob_desc_.CopyAllFrom(blob_desc);
  }

  Argument(const Argument &other) {
    blob_id_.CopyFrom(other.blob_id_);
    blob_desc_.CopyAllFrom(other.blob_desc_);
  }

  Argument &operator=(const Argument &other) {
    blob_id_.CopyFrom(other.blob_id_);
    blob_desc_.CopyAllFrom(other.blob_desc_);
    return *this;
  }

  DataType data_type() const { return blob_desc_.data_type(); }

  Shape shape() const { return blob_desc_.shape(); }

  ShapeProto shape_proto() const {
    ShapeProto proto;
    blob_desc_.shape().ToProto(&proto);
    return proto;
  }

  std::string blob_name() const { return BlobName(blob_id_); }

  const LogicalBlobId &blob_id() const { return blob_id_; }

  const BlobDesc &blob_desc() const { return blob_desc_; }

  bool operator==(const Argument &rhs) const {
    return blob_id_ == rhs.blob_id_;
  }

  friend struct std::hash<oneflow::mola::Argument>;

 private:
  LogicalBlobId blob_id_;
  BlobDesc blob_desc_;
};

}  // namespace mola
}  // namespace oneflow

namespace std {
template <>
struct hash<oneflow::mola::Argument> {
  size_t operator()(const oneflow::mola::Argument &arg) const {
    return std::hash<oneflow::LogicalBlobId>()(arg.blob_id_);
  }
};
}  // namespace std

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_
