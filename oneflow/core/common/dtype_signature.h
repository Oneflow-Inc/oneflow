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
