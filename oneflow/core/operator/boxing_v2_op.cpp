#include "oneflow/core/operator/boxing_v2_op.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/tensor_partial_view.h"

namespace oneflow {

void BoxingV2Op::InitFromOpConf() {
  EnrollRepeatedInputBn("in", GetCustomizedBoxingConf().in_view_size(), false);
  EnrollOutputBn("out");
  VirtualInitFromOpConf();
}

LogicalBlobId BoxingV2Op::ibn2lbi(const std::string& input_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

LogicalBlobId BoxingV2Op::obn2lbi(const std::string& output_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

const BoxingV2Conf& BoxingV2Op::GetCustomizedBoxingConf() const {
  return GetMsgFromCustomizedConf<BoxingV2Conf>("boxing_conf");
}

void BoxingV2Op::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const BoxingV2Conf& boxing_conf = GetCustomizedBoxingConf();
  const PbRpf<TensorPartialViewProto>& in_view_proto = boxing_conf.in_view();
  const TensorPartialViewProto& out_view_proto = boxing_conf.out_view();
  const BlobDesc* in_0 = GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  const DataType data_type = in_0->data_type();
  FOR_RANGE(int64_t, i, 1, input_bns().size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    CHECK_EQ(in_i->data_type(), data_type);
  }
  FOR_RANGE(int64_t, i, 0, input_bns().size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    const TensorPartialView in_i_view(in_view_proto.Get(i));
    CHECK_EQ(in_i->shape(), in_i_view.shape());
  }
  const TensorPartialView out_view(out_view_proto);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(data_type);
  out->mut_shape() = out_view.shape();
  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

const PbMessage& BoxingV2CopyOp::GetCustomizedConf() const {
  return op_conf().boxing_v2_copy_conf();
}

const PbMessage& BoxingV2AddOp::GetCustomizedConf() const { return op_conf().boxing_v2_add_conf(); }

void BoxingV2AddOp::VirtualInitFromOpConf() { EnrollFwBufBn("buf"); }

void BoxingV2AddOp::VirtualInferBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("buf") = *GetBlobDesc4BnInOp("out");
}

REGISTER_OP(OperatorConf::kBoxingV2CopyConf, BoxingV2CopyOp);
REGISTER_OP(OperatorConf::kBoxingV2AddConf, BoxingV2AddOp);

}  // namespace oneflow
