#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/op_registration.h"

namespace oneflow {

class UserOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOp);
  UserOp() = default;
  ~UserOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_user_conf());
    for (const auto& pair : op_conf().user_conf().input()) {
      EnrollRepeatedInputBn(pair.first, pair.second.s_size());
    }
    for (const auto& pair : op_conf().user_conf().output()) {
      EnrollRepeatedOutputBn(pair.first, pair.second.s_size());
    }
    EnrollTmpBn("tmp_buffer");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().user_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const user_op::OpRegistrationVal* val =
        user_op::LookUpInOpRegistry(op_conf().user_conf().op_type_name());
    CHECK_OR_RETURN(val != nullptr)
        << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in op registry!";
    // default method set other attribute instead of Shape and Dtype (such as data_id, is_dynamic)
    // set out blob desc other attr as first input blob desc (if has)
    // TODO(ChengCheng): infer other attribute in blob desc
    if (input_bns().empty() == false) {
      BlobDesc* first_in_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
      for (const std::string& obn : output_bns()) {
        *GetBlobDesc4BnInOp(obn) = *first_in_blob_desc;
      }
    }

    // infer Shape
    auto GetShape4ArgNameAndIndex = [&](const std::string& bn, int32_t index) -> Shape* {
      BlobDesc* blob = GetBlobDesc4BnInOp(GenRepeatedBn(bn, index));
      if (blob) { return &(blob->mut_shape()); }
      return nullptr;
    };
    JUST(val->shape_infer_fn(GetShape4ArgNameAndIndex));

    // infer Dtype
    HashMap<std::string, DataType> bn_in_op2data_type;
    auto GetDtype4ArgNameAndIndex = [&](const std::string& bn, int32_t index) -> DataType* {
      BlobDesc* blob = GetBlobDesc4BnInOp(GenRepeatedBn(bn, index));
      if (blob) { return &bn_in_op2data_type[GenRepeatedBn(bn, index)]; }
      return nullptr;
    };
    JUST(val->dtype_infer_fn(GetDtype4ArgNameAndIndex));
    for (const auto& pair : bn_in_op2data_type) {
      GetBlobDesc4BnInOp(pair.first)->set_data_type(pair.second);
    }

    return Maybe<void>::Ok();
  }

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override {
    auto pair = GenUnRepeatedBn(input_bn);
    return GenLogicalBlobId(op_conf().user_conf().input().at(pair.first).s(pair.second));
  }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override {
    auto pair = GenUnRepeatedBn(output_bn);
    return GenLogicalBlobId(op_conf().user_conf().output().at(pair.first).s(pair.second));
  }
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    // TODO(ChengCheng): default func for infer batch axis, need to DELETE
    OptInt64* batch_axis = nullptr;
    for (const std::string& ibn : input_bns()) {
      if (BatchAxis4BnInOp(ibn)->has_value()) {
        batch_axis = BatchAxis4BnInOp(ibn);
        break;
      }
    }
    if (batch_axis) {
      for (const std::string& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *batch_axis; }
    }
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    // TODO
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kUserConf, UserOp);

}  // namespace oneflow
