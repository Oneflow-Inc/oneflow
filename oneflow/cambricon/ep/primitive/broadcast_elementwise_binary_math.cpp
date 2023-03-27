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
#include "oneflow/cambricon/ep/primitive/broadcast_elementwise_binary.h"
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/primitive/type_seq.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace mlu {

namespace {
template<typename T>
std::unique_ptr<ep::primitive::Fill> NewFillPrimitive() {
  return ep::primitive::NewPrimitive<ep::primitive::FillFactory>(DeviceType::kMLU,
                                                                 GetDataType<T>::value);
}

std::unique_ptr<ep::primitive::Cast> NewCastPrimitive(DataType from, DataType to) {
  return ep::primitive::NewPrimitive<ep::primitive::CastFactory>(DeviceType::kMLU, from, to);
}
}  // namespace

template<BinaryOp op>
cnnlOpTensorDesc_t GetCnnlOpTensorType();

#define INSTANCE_GET_CNNL_OP_TENSOR_TYPE(op, cnnl_op) \
  template<>                                          \
  cnnlOpTensorDesc_t GetCnnlOpTensorType<op>() {      \
    return cnnl_op;                                   \
  }

INSTANCE_GET_CNNL_OP_TENSOR_TYPE(BinaryOp::kAdd, CNNL_OP_TENSOR_ADD);
INSTANCE_GET_CNNL_OP_TENSOR_TYPE(BinaryOp::kSub, CNNL_OP_TENSOR_SUB);
INSTANCE_GET_CNNL_OP_TENSOR_TYPE(BinaryOp::kMul, CNNL_OP_TENSOR_MUL);
#undef INSTANCE_GET_CNNL_OP_TENSOR_TYPE

union CnnlOpTensorScale {
  int i;
  float f;
};

template<typename T>
CnnlOpTensorScale GetCnnlOpTensorScale(double scale) {
  CnnlOpTensorScale cnnl_scale;
  if (IsIntegralDataType(GetDataType<T>::value)) {
    cnnl_scale.i = static_cast<int>(scale);
  } else {
    cnnl_scale.f = static_cast<float>(scale);
  }
  return cnnl_scale;
}

template<BinaryOp op, typename T>
struct BinaryMathImpl {
  void operator()(Stream* stream, cnnlDataType_t cnnl_dtype, cnnlTensorDescriptor_t src0_desc,
                  const void* src0, cnnlTensorDescriptor_t src1_desc, const void* src1,
                  cnnlTensorDescriptor_t dst_desc, void* dst) const {
    auto cnnl_op_tensor_type = GetCnnlOpTensorType<op>();
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(stream->As<ep::MluStream>()->cnnl_handle(),
                                               src0_desc, src1_desc, dst_desc, &workspace_size));
    CnnlWorkspace workspace(stream->As<ep::MluStream>(), workspace_size);

    CnnlOpTensorDescriptor op_tensor_desc;
    op_tensor_desc.set(cnnl_op_tensor_type, cnnl_dtype, CNNL_NOT_PROPAGATE_NAN);

    auto alpha = GetCnnlOpTensorScale<T>(1);
    auto beta = GetCnnlOpTensorScale<T>(0);
    OF_CNNL_CHECK(cnnlOpTensor(stream->As<ep::MluStream>()->cnnl_handle(), op_tensor_desc.desc(),
                               &alpha, src0_desc, src0, &alpha, src1_desc, src1, workspace.dptr(),
                               workspace_size, &beta, dst_desc, dst));
  }
};

template<typename T>
struct BinaryMathImpl<BinaryOp::kDiv, T> {
  void operator()(Stream* stream, cnnlDataType_t cnnl_dtype, cnnlTensorDescriptor_t src0_desc,
                  const void* src0, cnnlTensorDescriptor_t src1_desc, const void* src1,
                  cnnlTensorDescriptor_t dst_desc, void* dst) {
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetDivWorkspaceSize(stream->As<ep::MluStream>()->cnnl_handle(), src0_desc,
                                          src1_desc, dst_desc, &workspace_size));
    CnnlWorkspace workspace(stream->As<ep::MluStream>(), workspace_size);

    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
    OF_CNNL_CHECK(cnnlDiv_v2(stream->As<ep::MluStream>()->cnnl_handle(), prefer, src0_desc, src0,
                             src1_desc, src1, workspace.dptr(), workspace_size, dst_desc, dst));
  }
};

template<typename T>
struct BinaryMathImpl<BinaryOp::kPow, T> {
  void operator()(Stream* stream, cnnlDataType_t cnnl_dtype, cnnlTensorDescriptor_t src0_desc,
                  const void* src0, cnnlTensorDescriptor_t src1_desc, const void* src1,
                  cnnlTensorDescriptor_t dst_desc, void* dst) {
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetPowWorkspaceSize(stream->As<ep::MluStream>()->cnnl_handle(), src0_desc,
                                          src1_desc, dst_desc, &workspace_size));
    CnnlWorkspace workspace(stream->As<ep::MluStream>(), workspace_size);
    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
    OF_CNNL_CHECK(cnnlPow(stream->As<ep::MluStream>()->cnnl_handle(), prefer, src0_desc, src0,
                          src1_desc, src1, workspace.dptr(), workspace_size, dst_desc, dst));
  }
};

template<BinaryOp op, typename T>
class BinaryMath : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryMath);
  BinaryMath() : cnnl_dtype_(ConvertToCnnlDataType(GetDataType<T>::value)) {
    primitive_fill_ = NewFillPrimitive<T>();
    CHECK_OR_THROW(primitive_fill_) << "fill primitive is not available.";
  }
  ~BinaryMath() override = default;

  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1, void* dst) {
    std::vector<int64_t> dst_dims;
    if (num_src0_dims > num_src1_dims) {
      dst_dims = ComputeBroadcastShape(num_src1_dims, src1_dims, num_src0_dims, src0_dims);
    } else {
      dst_dims = ComputeBroadcastShape(num_src0_dims, src0_dims, num_src1_dims, src1_dims);
    }

    DataType compute_dtype = GetBinaryComputeDataType<op, T>();
    if (compute_dtype != GetDataType<T>::value) {
      CnnlWorkspace cast_workspace(stream->As<ep::MluStream>());
      int element_size = GetSizeOfDataType(compute_dtype);
      int64_t src0_count = ComputeElementCount(num_src0_dims, src0_dims) * element_size;
      int64_t src1_count = ComputeElementCount(num_src1_dims, src1_dims) * element_size;
      int64_t dst_count = ComputeElementCount(dst_dims.size(), dst_dims.data()) * element_size;
      cast_workspace.resize(src0_count + src1_count + dst_count);

      char* cast_workspace_dptr = reinterpret_cast<char*>(cast_workspace.dptr());

      auto cast_input = NewCastPrimitive(GetDataType<T>::value, compute_dtype);
      cast_input->Launch(stream, src0, cast_workspace_dptr, src0_count / element_size);
      cast_input->Launch(stream, src1, cast_workspace_dptr + src0_count, src1_count / element_size);

      auto cnnl_compute_dtype = ConvertToCnnlDataType(compute_dtype);
      CnnlTensorDescriptor src0_desc, src1_desc, dst_desc;
      src0_desc.set(num_src0_dims, src0_dims, cnnl_compute_dtype);
      src1_desc.set(num_src1_dims, src1_dims, cnnl_compute_dtype);

      dst_desc.set(dst_dims.size(), dst_dims.data(), cnnl_compute_dtype);
      void* dst_tmp = cast_workspace_dptr + src0_count + src1_count;
      BinaryMathImpl<op, T>()(stream, cnnl_compute_dtype, src0_desc.desc(), cast_workspace_dptr,
                              src1_desc.desc(), cast_workspace_dptr + src0_count, dst_desc.desc(),
                              dst_tmp);

      auto cast_output = NewCastPrimitive(compute_dtype, GetDataType<T>::value);
      cast_output->Launch(stream, dst_tmp, dst, dst_count / element_size);
    } else {
      CnnlTensorDescriptor src0_desc, src1_desc, dst_desc;
      src0_desc.set(num_src0_dims, src0_dims, cnnl_dtype_);
      src1_desc.set(num_src1_dims, src1_dims, cnnl_dtype_);
      dst_desc.set(dst_dims.size(), dst_dims.data(), cnnl_dtype_);
      BinaryMathImpl<op, T>()(stream, cnnl_dtype_, src0_desc.desc(), src0, src1_desc.desc(), src1,
                              dst_desc.desc(), dst);
    }
  }

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) {
    DataType input_dtype = GetDataType<T>::value;
    CnnlWorkspace temp(stream->As<ep::MluStream>(), GetSizeOfDataType(input_dtype));
    primitive_fill_->Launch(stream, temp.dptr(), src0, 1);

    int64_t src0_dims[1] = {1};
    Launch(stream, 1, src0_dims, temp.dptr(), num_src1_dims, src1_dims, src1, dst);
  }

  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) {
    DataType input_dtype = GetDataType<T>::value;
    CnnlWorkspace temp(stream->As<ep::MluStream>(), GetSizeOfDataType(input_dtype));
    primitive_fill_->Launch(stream, temp.dptr(), src1, 1);

    int64_t src1_dims[1] = {1};
    Launch(stream, num_src0_dims, src0_dims, src0, 1, src1_dims, temp.dptr(), dst);
  }

 private:
  cnnlDataType_t cnnl_dtype_;
  std::unique_ptr<ep::primitive::Fill> primitive_fill_;
};

#define INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op, src_data_type_pair) \
  template<>                                                                                   \
  std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary<                   \
      binary_op, OF_PP_PAIR_FIRST(src_data_type_pair), OF_PP_PAIR_FIRST(src_data_type_pair)>(  \
      Scalar attr0, Scalar attr1) {                                                            \
    return std::unique_ptr<BroadcastElementwiseBinary>(                                        \
        new BinaryMath<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair)>);                      \
  }

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                 MLU_BINARY_MATH_OP_SEQ, MLU_PRIMITIVE_ALL_TYPE_SEQ);

#undef INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY

}  // namespace mlu
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
