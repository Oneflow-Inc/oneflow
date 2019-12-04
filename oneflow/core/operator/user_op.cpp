#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/blob_def.h"
#include "oneflow/core/framework/infer_util.h"

namespace oneflow {

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
    const user_op::OpRegistrationVal* val =
        user_op::LookUpInOpRegistry(op_conf().user_conf().op_type_name());
    CHECK_OR_RETURN(val != nullptr)
        << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in op registry!";
    JUST(val->get_sbp_fn(LogicalBlobDesc4Ibn, sbp_sig_list));
    return Maybe<void>::Ok();
  }
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

class UserOpKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit UserOpKernelRegContext(const UserOp* user_op,
                                  std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx)
      : user_op::KernelRegContext(user_op::UserOpConfWrapper(user_op->op_conf())) {
    const auto& op_conf = user_op->op_conf();
    CHECK(op_conf.has_user_conf());

    device_ = op_conf.device_type();
    parallel_ctx_ = parallel_ctx;

    {
      BlobDesc* first_blob_desc =
          FindValidBlobDescOfBnsInOp(GetBlobDesc4BnInOp, user_op->input_bns());
      if (!first_blob_desc) {
        first_blob_desc = FindValidBlobDescOfBnsInOp(GetBlobDesc4BnInOp, user_op->output_bns());
      }
      if (first_blob_desc) { data_type_ = first_blob_desc->data_type(); }
    }

    {
#define INSERT_TO_ARG2TENSOR_DESC(prefix)                                                   \
  for (const auto& bn : user_op->prefix##_bns()) {                                          \
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);                                     \
    if (!blob_desc) { continue; }                                                           \
    arg2tensor_desc_.emplace(GenUnRepeatedBn(bn),                                           \
                             user_op::BlobDef(blob_desc->shape(), blob_desc->data_type())); \
  }

      INSERT_TO_ARG2TENSOR_DESC(input)
      INSERT_TO_ARG2TENSOR_DESC(output)
      INSERT_TO_ARG2TENSOR_DESC(tmp)

#undef INSERT_TO_ARG2TENSOR_DESC
    }
  }
  ~UserOpKernelRegContext() = default;

  DeviceType device() const override { return device_; }
  DataType data_type() const override { return data_type_; }
  const ParallelContext& parallel_ctx() const override { return *parallel_ctx_; }
  const user_op::BlobDef* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                     int32_t index) const override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; }
    return &(it->second);
  }

 private:
  DeviceType device_;
  DataType data_type_;
  const ParallelContext* parallel_ctx_;
  HashMap<std::pair<std::string, int32_t>, user_op::BlobDef> arg2tensor_desc_;
};

void UserOp::InitFromOpConf() {
  CHECK(op_conf().has_user_conf());
  for (const auto& pair : op_conf().user_conf().input()) {
    EnrollRepeatedInputBn(pair.first, pair.second.s_size());
  }
  for (const auto& pair : op_conf().user_conf().output()) {
    EnrollRepeatedOutputBn(pair.first, pair.second.s_size());
  }
  EnrollTmpBn(GenRepeatedBn("tmp_buffer", 0));
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

  // construct InferContext
  const auto& user_op_conf = op_conf().user_conf();
  user_op::Arg2BlobDef arg2blob_def;
  for (auto it = user_op_conf.input().begin(); it != user_op_conf.input().end(); ++it) {
    const std::string& arg_name = it->first;
    for (int32_t i = 0; i < it->second.s_size(); ++i) {
      BlobDesc* blob = GetBlobDesc4BnInOp(GenRepeatedBn(arg_name, i));
      arg2blob_def.emplace(std::make_pair(arg_name, i),
                           user_op::BlobDef(blob->shape(), blob->data_type()));
    }
  }
  for (auto it = user_op_conf.output().begin(); it != user_op_conf.output().end(); ++it) {
    const std::string& arg_name = it->first;
    for (int32_t i = 0; i < it->second.s_size(); ++i) {
      BlobDesc* blob = GetBlobDesc4BnInOp(GenRepeatedBn(arg_name, i));
      arg2blob_def.emplace(std::make_pair(arg_name, i),
                           user_op::BlobDef(blob->shape(), blob->data_type()));
    }
  }

  user_op::InferContext infer_ctx(user_op::UserOpConfWrapper(op_conf()), std::move(arg2blob_def));

  JUST(val->shape_infer_fn(&infer_ctx));
  JUST(val->dtype_infer_fn(&infer_ctx));
  for (const auto& pair : infer_ctx.outputs()) {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn(pair.first, pair.second));
    out_blob_desc->set_data_type(*(infer_ctx.Dtype4ArgNameAndIndex(pair.first, pair.second)));
    out_blob_desc->mut_shape() = *(infer_ctx.Shape4ArgNameAndIndex(pair.first, pair.second));
  }

  // infer tmp buffer size must after infer out shape/dtype
  const user_op::KernelRegistrationVal* kernel_reg_val = user_op::LookUpInKernelRegistry(
      op_conf().user_conf().op_type_name(),
      UserOpKernelRegContext(this, GetBlobDesc4BnInOp, parallel_ctx));
  CHECK_OR_RETURN(kernel_reg_val != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in kernel registry !";

  size_t tmp_size = kernel_reg_val->infer_tmp_size_fn(infer_ctx);
  if (tmp_size > 0) {
    BlobDesc* tmp_buffer_blob = GetBlobDesc4BnInOp(GenRepeatedBn("tmp_buffer", 0));
    CHECK(tmp_buffer_blob != nullptr);
    tmp_buffer_blob->set_data_type(DataType::kChar);
    tmp_buffer_blob->mut_shape() = Shape({static_cast<int64_t>(tmp_size)});
  }
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
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);     \
    if (!blob_desc) { continue; }                           \
    blob_desc->ToProto(&proto);                             \
    (*user_conf->mutable_bn_in_op2blob_desc())[bn] = proto; \
  }

  BLOB_DESCS_TO_PROTO(input)
  BLOB_DESCS_TO_PROTO(output)
  BLOB_DESCS_TO_PROTO(tmp)

#undef BLOB_DESCS_TO_PROTO
}

REGISTER_OP(OperatorConf::kUserConf, UserOp);

}  // namespace oneflow
