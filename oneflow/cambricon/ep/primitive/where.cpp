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

void Expand(Stream* stream, cnnlDataType_t data_type, size_t src_ndim, const int64_t* src_dims,
            const void* src, size_t dst_ndim, const int64_t* dst_dims, void* dst) {
  CnnlTensorDescriptor src_desc, dst_desc;
  src_desc.set(src_ndim, src_dims, data_type);
  dst_desc.set(dst_ndim, dst_dims, data_type);
  OF_CNNL_CHECK(cnnlExpand(stream->As<ep::MluStream>()->cnnl_handle(), src_desc.desc(), src,
                           dst_desc.desc(), dst));
}

template<typename T, typename CondT>
class WhereImpl : public Where {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereImpl);
  explicit WhereImpl() = default;
  ~WhereImpl() override = default;

  void Launch(Stream* stream, size_t num_cond_dims, const int64_t* cond_dims, const void* cond,
              size_t num_x_dims, const int64_t* x_dims, const void* x, size_t num_y_dims,
              const int64_t* y_dims, const void* y, void* z) override {
    DataType cond_data_type = GetDataType<CondT>::value;
    CHECK_OR_THROW(cond_data_type == DataType::kBool || cond_data_type == DataType::kInt8)
        << "condition data type should be oneflow.bool or oneflow.int8.";
    cnnlDataType_t cnnl_cond_data_type, cnnl_xy_data_type;
    cnnl_cond_data_type = ConvertToCnnlDataType(cond_data_type);
    switch (sizeof(T)) {
      case 1: cnnl_xy_data_type = ConvertToCnnlDataType(kInt8); break;
      case 2: cnnl_xy_data_type = ConvertToCnnlDataType(kInt16); break;
      case 4: cnnl_xy_data_type = ConvertToCnnlDataType(kInt32); break;
      case 8: cnnl_xy_data_type = ConvertToCnnlDataType(kInt64); break;
      default: UNIMPLEMENTED_THEN_THROW();
    }
    // get z_desc
    int64_t max_element_count = 1;
    int64_t z_dims[kMaxNumDims];
    size_t max_dims = std::max(num_x_dims, num_y_dims);
    max_dims = std::max(max_dims, num_cond_dims);
    for (size_t i = 0; i < max_dims; ++i) {
      size_t cond_lpad = max_dims - num_cond_dims;
      size_t x_lpad = max_dims - num_x_dims;
      size_t y_lpad = max_dims - num_y_dims;
      int64_t cond_dim = (i < cond_lpad) ? 1 : cond_dims[i - cond_lpad];
      int64_t x_dim = (i < x_lpad) ? 1 : x_dims[i - x_lpad];
      int64_t y_dim = (i < y_lpad) ? 1 : y_dims[i - y_lpad];
      int64_t max_dim = std::max(x_dim, y_dim);
      max_dim = std::max(max_dim, cond_dim);
      z_dims[i] = max_dim;
      max_element_count *= max_dim;
    }

    // expand inputs to same shape since cnnlSelectV2 broadcasting may lead to incorrect result when
    // the calculation scale is large
    Shape z_shape(DimVector(z_dims, z_dims + max_dims));
    Shape cond_shape(DimVector(cond_dims, cond_dims + num_cond_dims));
    Shape x_shape(DimVector(x_dims, x_dims + num_x_dims));
    Shape y_shape(DimVector(y_dims, y_dims + num_y_dims));
    auto* mlu_stream = stream->As<ep::MluStream>();
    CnnlWorkspace workspace_cond(mlu_stream), workspace_x(mlu_stream), workspace_y(mlu_stream);
    if (cond_shape != z_shape) {
      workspace_cond.resize(max_element_count * GetSizeOfDataType(cond_data_type));
      Expand(stream, cnnl_cond_data_type, num_cond_dims, cond_dims, cond, max_dims, z_dims,
             workspace_cond.dptr());
      cond = workspace_cond.dptr();
    }
    if (x_shape != z_shape) {
      workspace_x.resize(max_element_count * sizeof(T));
      Expand(stream, cnnl_xy_data_type, num_x_dims, x_dims, x, max_dims, z_dims,
             workspace_x.dptr());
      x = workspace_x.dptr();
    }
    if (y_shape != z_shape) {
      workspace_y.resize(max_element_count * sizeof(T));
      Expand(stream, cnnl_xy_data_type, num_y_dims, y_dims, y, max_dims, z_dims,
             workspace_y.dptr());
      y = workspace_y.dptr();
    }
    CnnlTensorDescriptor condition_desc, x_desc, y_desc, z_desc;
    condition_desc.set(max_dims, z_dims, cnnl_cond_data_type);
    x_desc.set(max_dims, z_dims, cnnl_xy_data_type);
    y_desc.set(max_dims, z_dims, cnnl_xy_data_type);
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
