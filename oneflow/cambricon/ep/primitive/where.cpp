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

#include "oneflow/core/ep/include/primitive/where.h"
#include <cstdint>
#include <type_traits>
#include "cnnl.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/core/common/constant.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/ep/common/primitive/where.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/cambricon/common/mlu_util.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace {

template<typename T, typename CondT>
class WhereImpl : public Where {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereImpl);
  explicit WhereImpl() = default;
  ~WhereImpl() override = default;

  void Launch(Stream* stream, size_t num_cond_dims, const int64_t* cond_dims, const void* cond,
              size_t num_x_dims, const int64_t* x_dims, const void* x, size_t num_y_dims,
              const int64_t* y_dims, const void* y, void* z) override {
    cnnlDataType_t cnnl_cond_data_type, cnnl_xy_data_type;
    if (std::is_same_v<CondT, bool>) {
      cnnl_cond_data_type = ConvertToCnnlDataType(kBool);
      switch (sizeof(T)) {
        case 1: cnnl_xy_data_type = ConvertToCnnlDataType(kInt8); break;
        case 2: cnnl_xy_data_type = ConvertToCnnlDataType(kInt16); break;
        case 4: cnnl_xy_data_type = ConvertToCnnlDataType(kInt32); break;
        case 8: cnnl_xy_data_type = ConvertToCnnlDataType(kInt64); break;
        default: UNIMPLEMENTED_THEN_THROW();
      }
    } else {
      UNIMPLEMENTED_THEN_THROW();
    }

    CnnlTensorDescriptor condition_desc, x_desc, y_desc, z_desc;
    condition_desc.set(num_cond_dims, cond_dims, cnnl_cond_data_type);
    x_desc.set(num_x_dims, x_dims, cnnl_xy_data_type);
    y_desc.set(num_y_dims, y_dims, cnnl_xy_data_type);
    // get z_desc
    int64_t z_dims[kMaxNumDims];
    size_t max_dims = std::max(num_x_dims, num_y_dims);
    for (int i = max_dims - 1; i >= 0; i--) {
      if (num_x_dims > i && num_y_dims > i) {
        z_dims[i] = std::max(x_dims[i], y_dims[i]);
      } else if (num_x_dims > i) {
        z_dims[i] = x_dims[i];
      } else {
        z_dims[i] = y_dims[i];
      }
    }
    z_desc.set(max_dims, z_dims, cnnl_xy_data_type);
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetSelectV2WorkspaceSize(stream->As<ep::MluStream>()->cnnl_handle(),
                                               condition_desc.desc(), x_desc.desc(), y_desc.desc(),
                                               &workspace_size));
    CnnlWorkspace workspace(stream->As<ep::MluStream>(), workspace_size);
    OF_CNNL_CHECK(cnnlSelectV2(stream->As<ep::MluStream>()->cnnl_handle(), condition_desc.desc(),
                               cond, x_desc.desc(), x, y_desc.desc(), y, workspace.dptr(),
                               workspace_size, z_desc.desc(), z));
  }
};

class WhereFactoryImpl : public WhereFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereFactoryImpl);
  WhereFactoryImpl() = default;
  ~WhereFactoryImpl() override = default;

  std::unique_ptr<Where> New(DataType cond_type, DataType data_type, size_t max_num_dims) override {
    return NewWhere<WhereImpl>(cond_type, data_type, max_num_dims);
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kMLU, WhereFactory, WhereFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
