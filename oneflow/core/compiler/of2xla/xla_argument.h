#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_

#include "absl/strings/str_cat.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

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

  std::string blob_name() const {
      return absl::StrCat(blob_id_.op_name(), "/", blob_id_.blob_name());
  }

 private:
  friend inline bool operator<(const Argument &lhs, const Argument &rhs);
  friend inline bool operator!=(const Argument &lhs, const Argument &rhs);
  friend inline bool operator==(const Argument &lhs, const Argument &rhs);
  friend struct std::hash<oneflow::mola::Argument>;

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
    return std::hash<oneflow::LogicalBlobId>()(arg.blob_id_);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_ARGUMENT_H_
