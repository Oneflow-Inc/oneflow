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
#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_PAD_H_
#define ONEFLOW_CORE_PRIMITIVE_COMMON_PAD_H_

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace ep {

namespace primitive {

template<size_t num_dims, typename IndexType>
struct PadParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  IndexType padding_before[num_dims];
  IndexType padding_after[num_dims];
  IndexType out_size[num_dims];
  IndexType elem_cnt{};
  const void* src{};
  void* dst{};
};

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_PAD_H_
