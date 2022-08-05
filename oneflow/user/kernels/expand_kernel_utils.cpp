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

#include "oneflow/user/kernels/expand_kernel_utils.h"

namespace oneflow {

Maybe<void> getOutShapeAndStride(const std::vector<int32_t>& in_shape,
                                 const std::vector<int32_t>& expand_shape,
                                 std::vector<int32_t>& out_shape, std::vector<int32_t>& stride) {
  // NOTE(Liang Depeng): compute the input stride.
  std::vector<int32_t> original_stride(in_shape.size(), 1);
  for (int i = in_shape.size() - 2; i >= 0; --i) {
    original_stride[i] = in_shape[i + 1] * original_stride[i + 1];
  }

  // NOTE(Liang Depeng): compute the output stride and shape.
  out_shape.resize(expand_shape.size());
  stride.resize(expand_shape.size());
  int shift = out_shape.size() - in_shape.size();
  for (int i = out_shape.size() - 1; i >= 0; --i) {
    int index = i - shift;
    if (index >= 0) {
      if (expand_shape[i] == -1 || expand_shape[i] == in_shape[index]) {
        out_shape[i] = in_shape[index];
        stride[i] = original_stride[index];
      } else {
        CHECK_OR_RETURN(expand_shape[i] >= 0 && in_shape[index] == 1) << "Invalid expand shape ";
        out_shape[i] = expand_shape[i];
        stride[i] = 0;
      }
    } else {
      CHECK_GE_OR_RETURN(expand_shape[i], 0) << "Invalid expand shape ";
      out_shape[i] = expand_shape[i];
      if (expand_shape[i] == 1 && i < out_shape.size() - 1) {
        stride[i] = stride[i + 1];
      } else {
        stride[i] = 0;
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> getOutShapeAndStrideForFp(const std::vector<int32_t>& in_shape,
                                      const std::vector<int32_t>& logical_expand_shape,
                                      std::vector<int32_t>& out_shape,
                                      std::vector<int32_t>& stride) {
  std::vector<int32_t> expand_shape;
  expand_shape.resize(logical_expand_shape.size());
  const int offset = logical_expand_shape.size() - in_shape.size();
  // NOTE(Liang Depeng): compute the correct expand shape according to the actual input shape.
  for (int i = 0; i < logical_expand_shape.size(); ++i) {
    if (i < offset) {
      expand_shape[i] = logical_expand_shape[i];
    } else {
      expand_shape[i] = in_shape[i - offset] == 1
                            ? logical_expand_shape[i]
                            : std::min(logical_expand_shape[i], in_shape[i - offset]);
    }
  }
  JUST(getOutShapeAndStride(in_shape, expand_shape, out_shape, stride));
  return Maybe<void>::Ok();
}

Maybe<void> getOutShapeAndStrideForBp(const std::vector<int32_t>& logical_out_shape,
                                      const std::vector<int32_t>& logical_expand_shape,
                                      const std::vector<int32_t>& in_shape,
                                      std::vector<int32_t>& out_shape,
                                      std::vector<int32_t>& stride) {
  std::vector<int32_t> expand_shape;
  expand_shape.resize(logical_expand_shape.size());
  // NOTE(Liang Depeng): compute the correct expand shape according to the actual input shape.
  for (int i = 0; i < logical_expand_shape.size(); ++i) {
    expand_shape[i] = logical_expand_shape[i] == -1
                          ? in_shape[i]
                          : std::min(logical_expand_shape[i], in_shape[i]);
  }
  // NOTE(Liang Depeng): compute the correct output shape.
  const int offset = logical_expand_shape.size() - logical_out_shape.size();
  out_shape.resize(logical_out_shape.size());
  for (int i = 0; i < logical_out_shape.size(); ++i) {
    out_shape[i] = std::min(expand_shape[i + offset], logical_out_shape[i]);
  }
  std::vector<int32_t> duplicated_in_shape;
  JUST(getOutShapeAndStride(out_shape, expand_shape, duplicated_in_shape, stride));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
