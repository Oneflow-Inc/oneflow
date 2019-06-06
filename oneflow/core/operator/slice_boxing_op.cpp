#include "oneflow/core/operator/slice_boxing_op.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

void SliceBoxingOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in", GetCustomizedBoxingConf().in_slice_size(), false);
  EnrollOutputBn("out");
  VirtualInitFromOpConf();
}

LogicalBlobId SliceBoxingOp::ibn2lbi(const std::string& input_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

LogicalBlobId SliceBoxingOp::obn2lbi(const std::string& output_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

const SliceBoxingConf& SliceBoxingOp::GetCustomizedBoxingConf() const {
  return GetMsgFromCustomizedConf<SliceBoxingConf>("slice_boxing_conf");
}

void SliceBoxingOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const SliceBoxingConf& slice_boxing_conf = GetCustomizedBoxingConf();
  const PbRpf<TensorSliceViewProto>& in_slice_proto = slice_boxing_conf.in_slice();
  const TensorSliceViewProto& out_slice_proto = slice_boxing_conf.out_slice();
  const BlobDesc* in_0 = GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  const DataType data_type = in_0->data_type();
  FOR_RANGE(int64_t, i, 1, input_bns().size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    CHECK_EQ(in_i->data_type(), data_type);
  }
  FOR_RANGE(int64_t, i, 0, input_bns().size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    const TensorSliceView in_i_slice(in_slice_proto.Get(i));
    CHECK_EQ(in_i->shape().elem_cnt(), in_i_slice.shape().elem_cnt());
  }
  const TensorSliceView out_slice(out_slice_proto);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(data_type);
  if (slice_boxing_conf.has_out_shape()) {
    const Shape out_shape(slice_boxing_conf.out_shape());
    CHECK_EQ(out_shape.elem_cnt(), out_slice.shape().elem_cnt());
    out->mut_shape() = out_shape;
  } else {
    out->mut_shape() = out_slice.shape();
  }
  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

const PbMessage& SliceBoxingCopyOp::GetCustomizedConf() const {
  return op_conf().slice_boxing_copy_conf();
}

const PbMessage& SliceBoxingAddOp::GetCustomizedConf() const {
  return op_conf().slice_boxing_add_conf();
}

void SliceBoxingAddOp::VirtualInitFromOpConf() { EnrollFwBufBn("buf"); }

void SliceBoxingAddOp::VirtualInferBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("buf") = *GetBlobDesc4BnInOp("out");
}

REGISTER_OP(OperatorConf::kSliceBoxingCopyConf, SliceBoxingCopyOp);
REGISTER_OP(OperatorConf::kSliceBoxingAddConf, SliceBoxingAddOp);

}  // namespace oneflow
