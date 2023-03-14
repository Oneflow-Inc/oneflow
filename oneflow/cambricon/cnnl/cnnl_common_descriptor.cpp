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
#include "oneflow/cambricon/cnnl/cnnl_common_descriptor.h"

#include "oneflow/core/common/throw.h"

namespace oneflow {

// modify tensor size and stride order based on
// channels_first to channels_last or channels_last_3d.
// which this is not same with pytorch original layout,
// this real layout is based on data storage real order.
// example: modify channels_first tensor dim to cnnl nhwc tensor desc.
//            N    C H W  -->   N    H W C
//          C*H*W  1 W C  --> C*H*W  W C 1
void convertShapeAndStride(std::vector<int>& shape_info, std::vector<int>& stride_info) {
  CHECK_EQ_OR_THROW(shape_info.size(), stride_info.size())
      << "shape size need equal to stride size.";
  const int dim = shape_info.size();
  std::vector<int> temp_shape_info(dim);
  std::vector<int> temp_stride_info(dim);
  temp_shape_info[0] = shape_info[0];
  temp_stride_info[0] = stride_info[0];
  for (size_t i = 0; i < dim - 1; ++i) {
    const int index = (i + 1) % (dim - 1) + 1;
    temp_shape_info[i + 1] = shape_info[index];
    temp_stride_info[i + 1] = stride_info[index];
  }
  shape_info.assign(temp_shape_info.begin(), temp_shape_info.end());
  stride_info.assign(temp_stride_info.begin(), temp_stride_info.end());
}

}  // namespace oneflow
