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

#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {
extern Maybe<std::string> (*PlacementToString)(Symbol<ParallelDesc> placement);
namespace one {
const std::string TensorMeta::DebugString() const {
  std::stringstream ss;
  ss << "shape=" << shape().ToString() << ", ";
  ss << "dtype=" << CHECK_JUST(DType::Get(dtype()))->name();
  return ss.str();
}
const std::string MirroredTensorMeta::DebugString() const {
  std::stringstream ss;
  ss << "MirroredTensorMeta(";
  ss << TensorMeta::DebugString() << ", ";
  ss << "device=\"" << device()->ToString() << "\"";
  ss << ")";
  return ss.str();
}
const std::string ConsistentTensorMeta::DebugString() const {
  std::stringstream ss;
  ss << "ConsistentTensorMeta(";
  ss << TensorMeta::DebugString() << ", ";
  ss << "placement=" << *CHECK_JUST(PlacementToString(parallel_desc())) << ", ";
  ss << "Sbp=(";
  for (int i = 0; i < nd_sbp()->sbp_parallel_size(); i++) {
    if (i < nd_sbp()->sbp_parallel_size() - 1) {
      ss << *CHECK_JUST(SbpToString(nd_sbp()->sbp_parallel(i))) << ", ";
    } else {
      ss << *CHECK_JUST(SbpToString(nd_sbp()->sbp_parallel(i))) << ")";
    }
  }
  ss << ")";
  return ss.str();
}
}  // namespace one
}  // namespace oneflow
