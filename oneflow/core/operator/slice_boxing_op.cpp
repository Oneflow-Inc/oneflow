#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

class SliceBoxingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingOp);
  SliceBoxingOp() = default;
  ~SliceBoxingOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 protected:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const;
  virtual void VirtualInferBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}
  virtual void VirtualInitFromOpConf(){};

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

class SliceBoxingCopyOp final : public SliceBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingCopyOp);
  SliceBoxingCopyOp() = default;
  ~SliceBoxingCopyOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  Symbol<OperatorConf> GetOpConfWithoutOpNameAndLbn() const override;
};

class SliceBoxingAddOp final : public SliceBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingAddOp);
  SliceBoxingAddOp() = default;
  ~SliceBoxingAddOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Symbol<OperatorConf> GetOpConfWithoutOpNameAndLbn() const override;
};

void SliceBoxingOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in", GetCustomizedBoxingConf().in_slice_size(), false);
  EnrollOutputBn("out");
  VirtualInitFromOpConf();
}

LogicalBlobId SliceBoxingOp::lbi4ibn(const std::string& input_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

LogicalBlobId SliceBoxingOp::lbi4obn(const std::string& output_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

const SliceBoxingConf& SliceBoxingOp::GetCustomizedBoxingConf() const {
  return GetMsgFromCustomizedConf<SliceBoxingConf>("slice_boxing_conf");
}

Maybe<void> SliceBoxingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
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
  return Maybe<void>::Ok();
}

const PbMessage& SliceBoxingCopyOp::GetCustomizedConf() const {
  return op_conf().slice_boxing_copy_conf();
}

Symbol<OperatorConf> SliceBoxingCopyOp::GetOpConfWithoutOpNameAndLbn() const {
  OperatorConf op_conf(this->op_conf());
  op_conf.set_name("");
  CHECK(op_conf.has_slice_boxing_copy_conf());
  auto* boxing_conf = op_conf.mutable_slice_boxing_copy_conf();
  LogicalBlobId empty_logical_blob_id{};
  *boxing_conf->mutable_slice_boxing_conf()->mutable_lbi() = empty_logical_blob_id;
  return SymbolOf(op_conf);
}

const PbMessage& SliceBoxingAddOp::GetCustomizedConf() const {
  return op_conf().slice_boxing_add_conf();
}

void SliceBoxingAddOp::VirtualInitFromOpConf() { EnrollTmpBn("buf"); }

void SliceBoxingAddOp::VirtualInferBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("buf") = *GetBlobDesc4BnInOp("out");
}

Symbol<OperatorConf> SliceBoxingAddOp::GetOpConfWithoutOpNameAndLbn() const {
  OperatorConf op_conf(this->op_conf());
  op_conf.set_name("");
  CHECK(op_conf.has_slice_boxing_add_conf());
  auto* boxing_conf = op_conf.mutable_slice_boxing_add_conf();
  LogicalBlobId empty_logical_blob_id{};
  *boxing_conf->mutable_slice_boxing_conf()->mutable_lbi() = empty_logical_blob_id;
  return SymbolOf(op_conf);
}

REGISTER_OP(OperatorConf::kSliceBoxingCopyConf, SliceBoxingCopyOp);
REGISTER_OP(OperatorConf::kSliceBoxingAddConf, SliceBoxingAddOp);

}  // namespace oneflow
