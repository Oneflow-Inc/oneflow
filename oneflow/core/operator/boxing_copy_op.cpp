#include "oneflow/core/operator/boxing_copy_op.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/tensor_partial_view.h"

namespace oneflow {

void BoxingCopyOpBase::InitFromOpConf() {
  const int64_t in_num =
      GetPbRpfFromPbMessage<TensorPartialViewProto>(GetCustomizedConf(), "in_view").size();
  EnrollRepeatedInputBn("in", in_num, false);
  EnrollOutputBn("out");
}

LogicalBlobId BoxingCopyOpBase::ibn2lbi(const std::string& input_bn) const {
  return op_conf().boxing_copy_conf().lbi();
}

LogicalBlobId BoxingCopyOpBase::obn2lbi(const std::string& output_bn) const {
  return op_conf().boxing_copy_conf().lbi();
}

void BoxingCopyOpBase::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const PbRpf<TensorPartialViewProto>& in_view_proto =
      GetPbRpfFromPbMessage<TensorPartialViewProto>(GetCustomizedConf(), "in_view");
  const TensorPartialViewProto& out_view_proto =
      GetValFromCustomizedConf<TensorPartialViewProto>("out_view");
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
}

const PbMessage& BoxingCopyOp::GetCustomizedConf() const { return op_conf().boxing_copy_conf(); }
const PbMessage& BoxingCopyAddOp::GetCustomizedConf() const {
  return op_conf().boxing_copy_add_conf();
}

REGISTER_OP(OperatorConf::kBoxingCopyConf, BoxingCopyOp);
REGISTER_OP(OperatorConf::kBoxingCopyAddConf, BoxingCopyAddOp);

}  // namespace oneflow
