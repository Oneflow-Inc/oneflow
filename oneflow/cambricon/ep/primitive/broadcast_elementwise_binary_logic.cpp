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
cnnlLogicOp_t GetCnnlLogicOp();

#define INSTANCE_GET_CNNL_LOGIC_OP(op, cnnl_op) \
  template<>                                    \
  cnnlLogicOp_t GetCnnlLogicOp<op>() {          \
    return cnnl_op;                             \
  }

OF_PP_FOR_EACH_TUPLE(INSTANCE_GET_CNNL_LOGIC_OP, MLU_CNNL_LOGICAL_OP_SEQ)

#undef INSTANCE_GET_CNNL_LOGIC_OP

template<BinaryOp op, typename Src, typename Dst>
class BinaryLogical : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryLogical);
  BinaryLogical()
      : logical_op_(GetCnnlLogicOp<op>()),
        src_dtype_(ConvertToCnnlDataType(GetDataType<Src>::value)),
        dst_dtype_(ConvertToCnnlDataType(GetDataType<Dst>::value)) {
    primitive_fill_ = NewFillPrimitive<Src>();
    CHECK_OR_THROW(primitive_fill_) << "fill primitive is not available.";
  }
  ~BinaryLogical() override = default;

  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1, void* dst) {
    CnnlTensorDescriptor src0_desc, src1_desc, dst_desc;
    std::vector<int64_t> dst_dims;
    if (num_src0_dims > num_src1_dims) {
      dst_dims = ComputeBroadcastShape(num_src1_dims, src1_dims, num_src0_dims, src0_dims);
    } else {
      dst_dims = ComputeBroadcastShape(num_src0_dims, src0_dims, num_src1_dims, src1_dims);
    }

    DataType compute_dtype = GetBinaryComputeDataType<op, Src>();
    CnnlWorkspace cast_workspace(stream->As<ep::MluStream>());

    if (compute_dtype != GetDataType<Src>::value) {
      int element_size = GetSizeOfDataType(compute_dtype);
      int64_t src0_count = ComputeElementCount(num_src0_dims, src0_dims) * element_size;
      int64_t src1_count = ComputeElementCount(num_src1_dims, src1_dims) * element_size;
      cast_workspace.resize(src0_count + src1_count);
      char* cast_workspace_dptr = reinterpret_cast<char*>(cast_workspace.dptr());

      auto cast_input = NewCastPrimitive(GetDataType<Src>::value, compute_dtype);
      cast_input->Launch(stream, src0, cast_workspace_dptr, src0_count / element_size);
      cast_input->Launch(stream, src1, cast_workspace_dptr + src0_count, src1_count / element_size);

      src0 = cast_workspace_dptr;
      src1 = cast_workspace_dptr + src0_count;
      auto cnnl_compute_dtype = ConvertToCnnlDataType(compute_dtype);
      src0_desc.set(num_src0_dims, src0_dims, cnnl_compute_dtype);
      src1_desc.set(num_src1_dims, src1_dims, cnnl_compute_dtype);
    } else {
      src0_desc.set(num_src0_dims, src0_dims, src_dtype_);
      src1_desc.set(num_src1_dims, src1_dims, src_dtype_);
    }
    dst_desc.set(dst_dims.size(), dst_dims.data(), dst_dtype_);

    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetLogicOpWorkspaceSize(stream->As<ep::MluStream>()->cnnl_handle(),
                                              src0_desc.desc(), src1_desc.desc(), dst_desc.desc(),
                                              &workspace_size));
    CnnlWorkspace workspace(stream->As<ep::MluStream>(), workspace_size);
    OF_CNNL_CHECK(cnnlLogicOp(stream->As<ep::MluStream>()->cnnl_handle(), logical_op_,
                              src0_desc.desc(), src0, src1_desc.desc(), src1, workspace.dptr(),
                              workspace_size, dst_desc.desc(), dst));
  }

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) {
    DataType input_dtype = GetDataType<Src>::value;
    CnnlWorkspace temp(stream->As<ep::MluStream>(), GetSizeOfDataType(input_dtype));
    primitive_fill_->Launch(stream, temp.dptr(), src0, 1);

    int64_t src0_dims[1] = {1};
    Launch(stream, 1, src0_dims, temp.dptr(), num_src1_dims, src1_dims, src1, dst);
  }

  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) {
    DataType input_dtype = GetDataType<Src>::value;
    CnnlWorkspace temp(stream->As<ep::MluStream>(), GetSizeOfDataType(input_dtype));
    primitive_fill_->Launch(stream, temp.dptr(), src1, 1);

    int64_t src1_dims[1] = {1};
    Launch(stream, num_src0_dims, src0_dims, src0, 1, src1_dims, temp.dptr(), dst);
  }

 private:
  cnnlLogicOp_t logical_op_;
  cnnlDataType_t src_dtype_;
  cnnlDataType_t dst_dtype_;
  std::unique_ptr<ep::primitive::Fill> primitive_fill_;
};

#define INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_LOGICAL_ENTRY(binary_op, src_data_type_pair, \
                                                                   dst_data_type_pair)            \
  template<>                                                                                      \
  std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary<                      \
      binary_op, OF_PP_PAIR_FIRST(src_data_type_pair), OF_PP_PAIR_FIRST(dst_data_type_pair)>(     \
      Scalar attr0, Scalar attr1) {                                                               \
    return std::unique_ptr<BroadcastElementwiseBinary>(                                           \
        new BinaryLogical<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair),                        \
                          OF_PP_PAIR_FIRST(dst_data_type_pair)>);                                 \
  }

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_LOGICAL_ENTRY,
                                 MLU_BINARY_LOGICAL_OP_SEQ, MLU_PRIMITIVE_ALL_TYPE_SEQ,
                                 MLU_PRIMITIVE_BOOL_TYPE_SEQ);

#undef INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_LOGICAL_ENTRY

}  // namespace mlu
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
