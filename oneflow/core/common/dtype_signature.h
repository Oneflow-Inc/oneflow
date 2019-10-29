#ifndef ONEFLOW_CORE_REGISTER_DTYPE_SIGNATURE_H_
#define ONEFLOW_CORE_REGISTER_DTYPE_SIGNATURE_H_

#include "oneflow/core/common/dtype_signature.pb.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

inline bool operator==(const DTypeSignature& lhs, const DTypeSignature& rhs) {
  return PbMd().Equals(lhs, rhs);
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::DTypeSignature> final {
  size_t operator()(const oneflow::DTypeSignature& dtype_signature) {
    std::string serialized;
    dtype_signature.SerializeToString(&serialized);
    return std::hash<std::string>()(serialized);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_REGISTER_DTYPE_SIGNATURE_H_
