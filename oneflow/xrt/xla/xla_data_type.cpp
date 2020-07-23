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
#include "glog/logging.h"

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/xrt/xla/xla_data_type.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace oneflow {
namespace xrt {
namespace mola {

xla::PrimitiveType DataTypeToPrimitiveType(DataType data_type) {
  switch (data_type) {
    case oneflow::kFloat: return xla::F32;
    case oneflow::kDouble: return xla::F64;
    case oneflow::kInt8: return xla::S8;
    case oneflow::kInt32: return xla::S32;
    case oneflow::kInt64: return xla::S64;
    case oneflow::kChar:
    case oneflow::kUInt8: return xla::U8;
    case oneflow::kFloat16: return xla::F16;
    default: {
      LOG(FATAL) << "Unsupported data type (" << data_type << ") in DataTypeToPrimitiveType";
      return xla::PRIMITIVE_TYPE_INVALID;
    }
  }
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
