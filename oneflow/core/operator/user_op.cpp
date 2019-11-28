#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/blob_info.h"

namespace oneflow {

class UserOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOp);
  UserOp() = default;
  ~UserOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().user_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    // TODO
    return Maybe<void>::Ok();
  }
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
  Maybe<void> InferTmpBufferBlobDesc(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;
};

namespace {

BlobDesc* FindValidBlobDescOfBnsInOp(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& bn_in_ops) {
  for (const std::string& bn_in_op : bn_in_ops) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc) { return blob_desc; }
  }
  return nullptr;
}

}  // namespace

void UserOp::InitFromOpConf() {
  CHECK(op_conf().has_user_conf());
  for (const auto& pair : op_conf().user_conf().input()) {
    EnrollRepeatedInputBn(pair.first, pair.second.s_size());
  }
  for (const auto& pair : op_conf().user_conf().output()) {
    EnrollRepeatedOutputBn(pair.first, pair.second.s_size());
  }
  EnrollTmpBn("tmp_buffer");
}

Maybe<void> UserOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const user_op::OpRegistrationVal* val =
      user_op::LookUpInOpRegistry(op_conf().user_conf().op_type_name());
  CHECK_OR_RETURN(val != nullptr) << "cannot find op_type: " << op_conf().user_conf().op_type_name()
                                  << " in op registry!";
  // default method set other attribute instead of Shape and Dtype (such as data_id, is_dynamic)
  // set out blob desc other attr as first input blob desc (if has)
  // TODO(ChengCheng): infer other attribute in blob desc
  BlobDesc* first_in_blob_desc = FindValidBlobDescOfBnsInOp(GetBlobDesc4BnInOp, input_bns());
  if (first_in_blob_desc) {
    for (const std::string& obn : output_bns()) { *GetBlobDesc4BnInOp(obn) = *first_in_blob_desc; }
  }

  // cclog:
  LOG(INFO) << "cclog: befor infer shape, in shape "
            << GetBlobDesc4BnInOp(SoleIbn())->shape().ToString() << " sole_ibn = " << SoleIbn();
  LOG(INFO) << "cclog: befor infer shape, out shape "
            << GetBlobDesc4BnInOp(SoleObn())->shape().ToString() << " sole_obn = " << SoleObn();

  // infer Shape
  auto GetShape4ArgNameAndIndex = [&](const std::string& bn, int32_t index) -> Shape* {
    BlobDesc* blob = GetBlobDesc4BnInOp(GenRepeatedBn(bn, index));
    if (blob) { return &(blob->mut_shape()); }
    return nullptr;
  };
  JUST(val->shape_infer_fn(GetShape4ArgNameAndIndex));

  // cclog:
  LOG(INFO) << "cclog: after infer shape, in shape "
            << GetBlobDesc4BnInOp(SoleIbn())->shape().ToString() << " sole_ibn = " << SoleIbn();
  LOG(INFO) << "cclog: after infer shape, out shape "
            << GetBlobDesc4BnInOp(SoleObn())->shape().ToString() << " sole_obn = " << SoleObn();

  // infer Dtype
  HashMap<std::string, DataType> bn_in_op2data_type;
  auto GetDtype4ArgNameAndIndex = [&](const std::string& bn, int32_t index) -> DataType* {
    std::string bn_in_op = GenRepeatedBn(bn, index);
    BlobDesc* blob = GetBlobDesc4BnInOp(bn_in_op);
    if (blob) { return &bn_in_op2data_type[bn_in_op]; }
    return nullptr;
  };
  JUST(val->dtype_infer_fn(GetDtype4ArgNameAndIndex));
  for (const auto& pair : bn_in_op2data_type) {
    GetBlobDesc4BnInOp(pair.first)->set_data_type(pair.second);
  }

  // infer tmp buffer size must after infer out shape/dtype
  JUST(InferTmpBufferBlobDesc(GetBlobDesc4BnInOp, parallel_ctx));
  return Maybe<void>::Ok();
}

LogicalBlobId UserOp::ibn2lbi(const std::string& input_bn) const {
  auto pair = GenUnRepeatedBn(input_bn);
  return GenLogicalBlobId(op_conf().user_conf().input().at(pair.first).s(pair.second));
}

LogicalBlobId UserOp::obn2lbi(const std::string& output_bn) const {
  auto pair = GenUnRepeatedBn(output_bn);
  return GenLogicalBlobId(op_conf().user_conf().output().at(pair.first).s(pair.second));
}

Maybe<void> UserOp::InferTmpBufferBlobDesc(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  user_op::BlobInfo4ArgNameAndIndexFn GetBlobInfo =
      [&](const std::string& arg_name, int32_t id) -> std::shared_ptr<user_op::BlobInfo> {
    BlobDesc* blob = GetBlobDesc4BnInOp(GenRepeatedBn(arg_name, id));
    if (blob) {
      return std::shared_ptr<user_op::BlobInfo>(
          new user_op::BlobInfo(blob->shape(), blob->data_type()));
    }
    return std::shared_ptr<user_op::BlobInfo>();
  };

  DataType data_type = DataType::kInvalidDataType;
  BlobDesc* first_blob_desc = FindValidBlobDescOfBnsInOp(GetBlobDesc4BnInOp, input_bns());
  if (!first_blob_desc) {
    first_blob_desc = FindValidBlobDescOfBnsInOp(GetBlobDesc4BnInOp, output_bns());
  }
  if (first_blob_desc) { data_type = first_blob_desc->data_type(); }

  user_op::KernelRegContext kernel_reg_ctx(op_conf().device_type(), data_type, *parallel_ctx,
                                           GetBlobInfo);
  const user_op::KernelRegistrationVal* kernel_reg_val =
      user_op::LookUpInKernelRegistry(op_conf().user_conf().op_type_name(), kernel_reg_ctx);
  CHECK_OR_RETURN(kernel_reg_val != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in kernel registry!";

  size_t tmp_size = kernel_reg_val->infer_tmp_size_fn(/*TODO(niuchong)*/);
  if (tmp_size > 0) {
    BlobDesc* tmp_buffer_blob = GetBlobDesc4BnInOp("tmp_buffer");
    CHECK(tmp_buffer_blob != nullptr);
    tmp_buffer_blob->set_data_type(DataType::kChar);
    tmp_buffer_blob->mut_shape() = Shape({static_cast<int64_t>(tmp_size)});
  }
  return Maybe<void>::Ok();
}

Maybe<void> UserOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
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

void UserOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  auto user_conf = kernel_conf->mutable_user_conf();
  *(user_conf->mutable_parallel_ctx()) = *parallel_ctx;
#define BLOB_DESCS_TO_PROTO(prefix)                         \
  for (const auto& bn : prefix##_bns()) {                   \
    BlobDescProto proto;                                    \
    GetBlobDesc4BnInOp(bn)->ToProto(&proto);                \
    (*user_conf->mutable_bn_in_op2blob_desc())[bn] = proto; \
  }

  BLOB_DESCS_TO_PROTO(input)
  BLOB_DESCS_TO_PROTO(output)
  BLOB_DESCS_TO_PROTO(tmp)

#undef BLOB_DESCS_TO_PROTO
}

REGISTER_OP(OperatorConf::kUserConf, UserOp);

}  // namespace oneflow
