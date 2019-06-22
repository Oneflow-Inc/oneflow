#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_

#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {
namespace mola {

class Argument {
 public:
  Argument() {}
  Argument(const LogicalBlobId &blob_id, const BlobDesc &blob_desc)
      : blob_id_(blob_id), blob_desc_(blob_desc) {}

  Argument(const Argument &other) : blob_id_(other.blob_id_),
                                    blob_desc_(other.blob_desc_) {}
 
  DataType data_type() const { return blob_desc_.data_type(); }

  Shape shape() const { return blob_desc_.shape(); }

  const LogicalBlobId &blob_id() const { return blob_id_; }
  const BlobDesc &blob_desc() const { return blob_desc_; }

 private:
  friend inline bool operator<(const Argument &lhs, const Argument &rhs);
  friend inline bool operator!=(const Argument &lhs, const Argument &rhs);
  friend inline bool operator==(const Argument &lhs, const Argument &rhs);

  LogicalBlobId blob_id_;
  BlobDesc blob_desc_;
};

inline bool operator<(const Argument &lhs, const Argument &rhs) {
  return lhs.blob_id_ < rhs.blob_id_;
}

inline bool operator!=(const Argument &lhs, const Argument &rhs) {
  return lhs.blob_id_ != rhs.blob_id_;
}

inline bool operator==(const Argument &lhs, const Argument &rhs) {
  return lhs.blob_id_ == rhs.blob_id_;
}

}  // namespace mola
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::mola::Argument> {
  size_t operator()(const oneflow::mola::Argument& arg) const {
    return std::hash<oneflow::LogicalBlobId>()(arg.blob_id());
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_
