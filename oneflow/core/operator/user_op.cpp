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
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/operator/user_op.h"
#include "oneflow/core/framework/infer_output_blob_time_shape_fn_context.h"
#include "oneflow/core/framework/infer_parallel_distribution_fn_context.h"

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

user_op::NaiveTensorDesc GenTensorDescFromBlobDesc(const BlobDesc* blob_desc) {
  user_op::NaiveTensorDesc tensor_desc;
  *tensor_desc.mut_shape() = blob_desc->shape();
  *tensor_desc.mut_data_type() = blob_desc->data_type();
  *tensor_desc.mut_is_dynamic() = blob_desc->is_dynamic();
  return tensor_desc;
}

}  // namespace

class UserOpKernelRegContext final : public user_op::KernelRegContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  explicit UserOpKernelRegContext(const UserOp* user_op,
                                  std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx)
      : user_op::KernelRegContext(user_op::UserOpConfWrapper(user_op->op_conf())) {
    const auto& op_conf = user_op->op_conf();
    CHECK(op_conf.has_user_conf());

    device_tag_ = op_conf.device_tag();
    device_type_ = CHECK_JUST(DeviceType4DeviceTag(device_tag_));
    parallel_ctx_ = parallel_ctx;

    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                           ArgVec* arg_vec) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg_vec->emplace_back(std::make_pair(it->first, i));
        }
      }
    };
    InitInOrOut(op_conf.user_conf().input(), &inputs_);
    InitInOrOut(op_conf.user_conf().output(), &outputs_);

    {
#define INSERT_TO_ARG2TENSOR_DESC(prefix)                                                \
  for (const auto& bn : user_op->prefix##_bns()) {                                       \
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);                                  \
    if (!blob_desc) { continue; }                                                        \
    arg2tensor_desc_.emplace(GenUnRepeatedBn(bn), GenTensorDescFromBlobDesc(blob_desc)); \
  }

      INSERT_TO_ARG2TENSOR_DESC(input)
      INSERT_TO_ARG2TENSOR_DESC(output)
      INSERT_TO_ARG2TENSOR_DESC(tmp)

#undef INSERT_TO_ARG2TENSOR_DESC
    }
  }
  ~UserOpKernelRegContext() = default;

  DeviceType device_type() const override { return device_type_; }
  const std::string& device_tag() const override { return device_tag_; }
  const ParallelContext& parallel_ctx() const override { return *parallel_ctx_; }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; }
    return &(it->second);
  }
  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }

 private:
  ArgVec inputs_;
  ArgVec outputs_;
  DeviceType device_type_;
  std::string device_tag_;
  const ParallelContext* parallel_ctx_;
  HashMap<std::pair<std::string, int32_t>, user_op::NaiveTensorDesc> arg2tensor_desc_;
};

class UserOpInferContext : public user_op::InferContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  UserOpInferContext(const UserOp* op, const ParallelContext* parallel_ctx, const JobDesc* job_desc,
                     const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp)
      : op_(op), parallel_ctx_(parallel_ctx), job_desc_(job_desc) {
    bn2logical_tensor_desc_.reset(new HashMap<std::string, user_op::NaiveTensorDesc>());
    auto InitTensorDesc = [&](const ArgVec& arg_vec, const PbRpf<std::string>& bns) {
      CHECK_EQ(arg_vec.size(), bns.size());
      for (int32_t i = 0; i < arg_vec.size(); ++i) {
        BlobDesc* blob = GetBlobDesc4BnInOp(bns.Get(i));
        CHECK_NOTNULL(blob);
        arg2tensor_desc_.emplace(arg_vec.at(i), GenTensorDescFromBlobDesc(blob));
      }
    };
    InitTensorDesc(op->inputs(), op->input_bns());
    InitTensorDesc(op->outputs(), op->output_bns());
  }
  ~UserOpInferContext() override = default;

  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                  int32_t index) override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; };
    return &(it->second);
  }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    const std::string bn = GenRepeatedBn(arg_name, index);
    const auto it = bn2logical_tensor_desc_->find(bn);
    if (it != bn2logical_tensor_desc_->end()) {
      return &it->second;
    } else {
      std::shared_ptr<const BlobDesc> blob_desc = CHECK_JUST(op_->GetLogicalBlobDesc4BnInOp(bn));
      bn2logical_tensor_desc_->emplace(bn, GenTensorDescFromBlobDesc(blob_desc.get()));
      return &(bn2logical_tensor_desc_->emplace(bn, GenTensorDescFromBlobDesc(blob_desc.get()))
                   .first->second);
    }
  }
  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; };
    return it->second.mut_shape();
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; };
    return it->second.mut_data_type();
  }
  bool* IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; };
    return it->second.mut_is_dynamic();
  }

  const ArgVec& inputs() const override { return op_->inputs(); }
  const ArgVec& outputs() const override { return op_->outputs(); }
  const ParallelContext& parallel_ctx() const override { return *parallel_ctx_; };
  const ParallelDesc& parallel_desc() const override {
    return *CHECK_JUST(op_->GetOpParallelDesc());
  };
  const JobDesc* job_desc() const override {
    CHECK_NOTNULL(job_desc_);
    return job_desc_;
  }

  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    CHECK_EQ(CHECK_JUST(op_->GetOpParallelDesc())->hierarchy()->NumAxes(), 1);
    const auto& bn2sbp = CHECK_JUST(op_->sbp_signature())->bn_in_op2sbp_parallel();
    std::string bn = GenRepeatedBn(arg_name, index);
    auto it = bn2sbp.find(bn);
    CHECK(it != bn2sbp.end());
    return it->second;
  }

  const ParallelDistribution& ParallelDistribution4ArgNameAndIndex(const std::string& arg_name,
                                                                   int32_t index) const override {
    const auto& bn2parallel_distribution =
        CHECK_JUST(op_->parallel_distribution_signature())->bn_in_op2parallel_distribution();
    std::string bn = GenRepeatedBn(arg_name, index);
    auto it = bn2parallel_distribution.find(bn);
    CHECK(it != bn2parallel_distribution.end());
    return it->second;
  }

  int64_t parallel_num() const override {
    return CHECK_JUST(op_->GetOpParallelDesc())->parallel_num();
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return op_->user_op_conf(); }

 private:
  const UserOp* op_;
  const ParallelContext* parallel_ctx_;
  const JobDesc* job_desc_;
  HashMap<std::pair<std::string, int32_t>, user_op::NaiveTensorDesc> arg2tensor_desc_;
  std::unique_ptr<HashMap<std::string, user_op::NaiveTensorDesc>> bn2logical_tensor_desc_;
};

class UserOpSbpContext : public user_op::SbpContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  UserOpSbpContext(const UserOp* op, SbpSignatureList* sbp_sig_list,
                   std::function<Maybe<const BlobDesc&>(const std::string&)> LogicalBlobDesc4Ibn)
      : op_(op), sbp_sig_list_(sbp_sig_list) {
    const auto& user_op_conf = op->op_conf().user_conf();
    for (auto it = user_op_conf.input().begin(); it != user_op_conf.input().end(); ++it) {
      const std::string& arg_name = it->first;
      for (int32_t i = 0; i < it->second.s_size(); ++i) {
        const BlobDesc* blob = &CHECK_JUST(LogicalBlobDesc4Ibn(GenRepeatedBn(arg_name, i)));
        arg2tensor_desc_.emplace(std::make_pair(arg_name, i), GenTensorDescFromBlobDesc(blob));
      }
    }
  }
  ~UserOpSbpContext() override = default;

  const user_op::TensorDesc& LogicalTensorDesc4InputArgNameAndIndex(
      const std::string& input_arg_name, int32_t index) const override {
    return arg2tensor_desc_.at(std::make_pair(input_arg_name, index));
  }
  const ArgVec& inputs() const override { return op_->inputs(); }
  const ArgVec& outputs() const override { return op_->outputs(); }
  const user_op::UserOpConfWrapper& user_op_conf() const override { return op_->user_op_conf(); }

  user_op::UserOpSbpSignatureBuilder NewBuilder() override {
    return user_op::UserOpSbpSignatureBuilder(sbp_sig_list_);
  }

  DeviceType device_type() const override { return op_->device_type(); }

  int64_t parallel_num() const override {
    return CHECK_JUST(op_->GetOpParallelDesc())->parallel_num();
  }

 private:
  const UserOp* op_;
  SbpSignatureList* sbp_sig_list_;
  HashMap<std::pair<std::string, int32_t>, user_op::NaiveTensorDesc> arg2tensor_desc_;
};

class UserOpInferSbpSignatureFnContext : public user_op::InferSbpSignatureFnContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  UserOpInferSbpSignatureFnContext(
      const UserOp* op, SbpSignature* signature, const SbpSignature& sbp_signature_conf,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn)
      : op_(op),
        signature_(signature),
        sbp_signature_conf_(sbp_signature_conf),
        sbp_infer_hint4ibn_fn_(std::move(SbpInferHint4Ibn)) {
    const auto& user_op_conf = op->op_conf().user_conf();
    for (const auto& it : user_op_conf.input()) {
      const std::string& arg_name = it.first;
      for (int32_t i = 0; i < it.second.s_size(); ++i) {
        auto hint = CHECK_JUST(sbp_infer_hint4ibn_fn_(GenRepeatedBn(arg_name, i)));
        arg2tensor_desc_.emplace(std::make_pair(arg_name, i),
                                 GenTensorDescFromBlobDesc(&hint->logical_blob_desc()));
        arg2sbp_parallel_hint_.emplace(std::make_pair(arg_name, i), hint->sbp_parallel());
      }
    }
  }
  ~UserOpInferSbpSignatureFnContext() override = default;

  const user_op::TensorDesc& LogicalTensorDesc4InputArgNameAndIndex(
      const std::string& input_arg_name, int32_t index) const override {
    return arg2tensor_desc_.at(std::make_pair(input_arg_name, index));
  }
  const ArgVec& inputs() const override { return op_->inputs(); }
  const ArgVec& outputs() const override { return op_->outputs(); }
  SbpSignature* mutable_sbp_signature() override { return signature_; }
  const SbpSignature& sbp_signature_conf() const override { return sbp_signature_conf_; }

  const SbpParallel& SbpParallelHint4InputArgNameAndIndex(const std::string& input_arg_name,
                                                          int32_t index) const override {
    return arg2sbp_parallel_hint_.at(std::make_pair(input_arg_name, index));
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return op_->user_op_conf(); }

  DeviceType device_type() const override { return op_->device_type(); }

  int64_t parallel_num() const override {
    return CHECK_JUST(op_->GetOpParallelDesc())->parallel_num();
  }

 private:
  const UserOp* op_;
  HashMap<std::pair<std::string, int32_t>, user_op::NaiveTensorDesc> arg2tensor_desc_;
  HashMap<std::pair<std::string, int32_t>, SbpParallel> arg2sbp_parallel_hint_;
  SbpSignature* signature_;
  SbpSignature sbp_signature_conf_;
  std::function<Maybe<const SbpInferHint*>(const std::string&)> sbp_infer_hint4ibn_fn_;
};

class UserOpInferOutputBlobTimeShapeFnContext : public user_op::InferOutputBlobTimeShapeFnContext {
 public:
  UserOpInferOutputBlobTimeShapeFnContext(
      const UserOp* op,
      const std::function<Maybe<const Shape>(const std::string&)>& GetTimeShape4BnInOp,
      Shape* output_blob_time_shape)
      : op_(op), output_blob_time_shape_(output_blob_time_shape) {
    for (const auto& it : op->op_conf().user_conf().input()) {
      const std::string& arg_name = it.first;
      for (int32_t i = 0; i < it.second.s_size(); ++i) {
        std::string ibn = GenRepeatedBn(arg_name, i);
        arg2time_shape_.emplace(std::make_pair(arg_name, i), *CHECK_JUST(GetTimeShape4BnInOp(ibn)));
      }
    }
  }
  ~UserOpInferOutputBlobTimeShapeFnContext() override = default;

  const Shape& TimeShape4InputArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return arg2time_shape_.at(std::make_pair(arg_name, index));
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return op_->user_op_conf(); }

  Shape* mut_output_blob_time_shape() override { return output_blob_time_shape_; };

 private:
  const UserOp* op_;
  HashMap<std::pair<std::string, int32_t>, Shape> arg2time_shape_;
  Shape* output_blob_time_shape_;
};

class UserOpInferParallelDistributionFnContext
    : public user_op::InferParallelDistributionFnContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;
  UserOpInferParallelDistributionFnContext(
      const UserOp* op, ParallelDistributionSignature* parallel_distribution_signature,
      const ParallelDistributionSignature& parallel_distribution_constraints,
      std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
          ParallelDistributionInferHint4Ibn)
      : op_(op),
        parallel_distribution_signature_(parallel_distribution_signature),
        parallel_distribution_constraints_(parallel_distribution_constraints),
        parallel_distribution_infer_hint4ibn_fn_(std::move(ParallelDistributionInferHint4Ibn)) {}
  ~UserOpInferParallelDistributionFnContext() override = default;

  const ParallelDistributionSignature& parallel_distribution_constraints() const override {
    return parallel_distribution_constraints_;
  }

  ParallelDistribution* ParallelDistribution4ArgNameAndIndex(const std::string& arg_name,
                                                             int32_t index) override {
    return &(*parallel_distribution_signature_
                  ->mutable_bn_in_op2parallel_distribution())[GenRepeatedBn(arg_name, index)];
  }

  const ParallelDistribution& ParallelDistributionHint4InputArgNameAndIndex(
      const std::string& arg_name, int32_t index) override {
    auto hint =
        CHECK_JUST(parallel_distribution_infer_hint4ibn_fn_(GenRepeatedBn(arg_name, index)));
    return hint->parallel_distribution();
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return op_->user_op_conf(); }

  int64_t parallel_num() const override {
    return CHECK_JUST(op_->GetOpParallelDesc())->parallel_num();
  }

  const Shape& parallel_hierarchy() override {
    return *(CHECK_JUST(op_->GetOpParallelDesc())->hierarchy());
  }

  const ArgVec& inputs() const override { return op_->inputs(); }
  const ArgVec& outputs() const override { return op_->outputs(); }

 private:
  const UserOp* op_;
  ParallelDistributionSignature* parallel_distribution_signature_;
  ParallelDistributionSignature parallel_distribution_constraints_;
  std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
      parallel_distribution_infer_hint4ibn_fn_;
};

void UserOp::InitFromOpConf() {
  CHECK(op_conf().has_user_conf());
  for (const auto& pair : op_conf().user_conf().input()) {
    EnrollRepeatedInputBn(pair.first, pair.second.s_size());
    for (int32_t i = 0; i < pair.second.s_size(); ++i) {
      inputs_.emplace_back(std::make_pair(pair.first, i));
    }
  }
  for (const auto& pair : op_conf().user_conf().output()) {
    EnrollRepeatedOutputBn(pair.first, pair.second.s_size());
    for (int32_t i = 0; i < pair.second.s_size(); ++i) {
      outputs_.emplace_back(std::make_pair(pair.first, i));
    }
  }
  EnrollTmpBn(GenRepeatedBn("tmp_buffer", 0));
  user_op_conf_.reset(new user_op::UserOpConfWrapper(shared_op_conf()));
  val_ =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_conf().user_conf().op_type_name());
  if (val_ != nullptr) {
    if (val_->input_arg_modify_fn) {
      user_op::GetInputArgModifier GetInputArgModifierFn =
          [&](const std::string& in_arg_name, int32_t in_arg_index) -> user_op::InputArgModifier* {
        std::string ibn = GenRepeatedBn(in_arg_name, in_arg_index);
        if (std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end()) {
          return MutInputBlobModifier4Ibn(ibn);
        }
        return nullptr;
      };
      val_->input_arg_modify_fn(GetInputArgModifierFn, *user_op_conf_);
    }
    if (val_->output_arg_modify_fn) {
      user_op::GetOutputArgModifier GetOutputArgModifierFn =
          [&](const std::string& out_arg_name,
              int32_t out_arg_index) -> user_op::OutputArgModifier* {
        std::string obn = GenRepeatedBn(out_arg_name, out_arg_index);
        if (std::find(output_bns().begin(), output_bns().end(), obn) != output_bns().end()) {
          return MutOutputBlobModifier4Obn(obn);
        }
        return nullptr;
      };
      val_->output_arg_modify_fn(GetOutputArgModifierFn, *user_op_conf_);
    }
  }
}

Maybe<void> UserOp::InferInternalBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  // tmp buffer size must be inferred after out shape/dtype
  UserOpInferContext infer_ctx(this, parallel_ctx, job_desc, GetBlobDesc4BnInOp);
  const user_op::OpKernelRegistryResult* kernel_reg_val =
      JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
          op_conf().user_conf().op_type_name(),
          UserOpKernelRegContext(this, GetBlobDesc4BnInOp, parallel_ctx)));
  CHECK_OR_RETURN(kernel_reg_val != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in kernel registry !";

  size_t tmp_size = kernel_reg_val->infer_tmp_size_fn(&infer_ctx);
  if (tmp_size > 0) {
    BlobDesc* tmp_buffer_blob = GetBlobDesc4BnInOp(GenRepeatedBn("tmp_buffer", 0));
    CHECK_NOTNULL_OR_RETURN(tmp_buffer_blob);
    tmp_buffer_blob->set_data_type(DataType::kChar);
    tmp_buffer_blob->mut_shape() = Shape({static_cast<int64_t>(tmp_size)});
  }
  return Maybe<void>::Ok();
}

Maybe<void> UserOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  CHECK_OR_RETURN(val_ != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in op registry!";
  // default method set output blob desc (such as Dtype, is_dynamic)
  // set out blob desc attr as first input blob desc (if has)
  BlobDesc* first_in_blob_desc = FindValidBlobDescOfBnsInOp(BlobDesc4BnInOp, input_bns());
  if (first_in_blob_desc) {
    for (const std::string& obn : output_bns()) {
      BlobDesc4BnInOp(obn)->CopyFrom(*first_in_blob_desc);
    }
  }

  UserOpInferContext infer_ctx(this, nullptr, nullptr, BlobDesc4BnInOp);

  CHECK_OR_RETURN(val_->data_type_infer_fn)
      << "No InferDataType function for " << val_->op_type_name;
  JUST(val_->data_type_infer_fn(&infer_ctx));
  JUST(val_->logical_tensor_desc_infer_fn(&infer_ctx));
  for (const auto& pair : infer_ctx.outputs()) {
    BlobDesc* out_blob_desc = BlobDesc4BnInOp(GenRepeatedBn(pair.first, pair.second));
    user_op::TensorDesc* tensor_desc =
        infer_ctx.TensorDesc4ArgNameAndIndex(pair.first, pair.second);
    out_blob_desc->set_data_type(tensor_desc->data_type());
    out_blob_desc->mut_shape() = tensor_desc->shape();
    out_blob_desc->set_is_dynamic(tensor_desc->is_dynamic());
  }
  return Maybe<void>::Ok();
}

Maybe<void> UserOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_OR_RETURN(val_ != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in op registry!";
  if (!val_->physical_tensor_desc_infer_fn) {
    return Operator::InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
  } else {
    // default method set output blob desc (such as Dtype, is_dynamic, is_tensor_list)
    // set out blob desc attr as first input blob desc (if has)
    BlobDesc* first_in_blob_desc = FindValidBlobDescOfBnsInOp(GetBlobDesc4BnInOp, input_bns());
    if (first_in_blob_desc) {
      for (const std::string& obn : output_bns()) {
        GetBlobDesc4BnInOp(obn)->CopyFrom(*first_in_blob_desc);
      }
    }
    UserOpInferContext infer_ctx(this, parallel_ctx, nullptr, GetBlobDesc4BnInOp);

    CHECK_OR_RETURN(val_->data_type_infer_fn)
        << "No InferDataType function for " << val_->op_type_name;
    JUST(val_->data_type_infer_fn(&infer_ctx));
    JUST(val_->physical_tensor_desc_infer_fn(&infer_ctx));
    for (const auto& pair : infer_ctx.outputs()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn(pair.first, pair.second));
      out_blob_desc->set_data_type(*(infer_ctx.Dtype4ArgNameAndIndex(pair.first, pair.second)));
      out_blob_desc->mut_shape() = *(infer_ctx.Shape4ArgNameAndIndex(pair.first, pair.second));
      out_blob_desc->set_is_dynamic(*infer_ctx.IsDynamic4ArgNameAndIndex(pair.first, pair.second));
    }
    return Maybe<void>::Ok();
  }
}

Maybe<void> UserOp::InferInplaceObn2Ibn(
    HashMap<std::string, std::string>* mut_inplace_obn2ibn,
    HashMap<std::string, std::string>* con_inplace_obn2ibn,
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  UserOpInferContext infer_ctx(this, parallel_ctx, nullptr, GetBlobDesc4BnInOp);
  const user_op::OpKernelRegistryResult* kernel_reg_val =
      JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
          op_conf().user_conf().op_type_name(),
          UserOpKernelRegContext(this, GetBlobDesc4BnInOp, parallel_ctx)));
  CHECK_OR_RETURN(kernel_reg_val != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in kernel registry !";
  HashSet<std::string> bn_in_op_unique_check;
  user_op::AddInplaceArgPair AddInplaceArgPairFn =
      [&](const std::string& out_arg_name, int32_t out_arg_index, const std::string& in_arg_name,
          int32_t in_arg_index, bool is_mutable) -> Maybe<void> {
    std::string ibn = GenRepeatedBn(in_arg_name, in_arg_index);
    std::string obn = GenRepeatedBn(out_arg_name, out_arg_index);
    if (is_mutable) {
      mut_inplace_obn2ibn->emplace(obn, ibn);
    } else {
      con_inplace_obn2ibn->emplace(obn, ibn);
    }
    CHECK_OR_RETURN(std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end())
        << "Cannot find input_arg_name : " << in_arg_name << " input_arg_index : " << in_arg_index
        << " in op_name: " << op_conf().name();
    CHECK_OR_RETURN(std::find(output_bns().begin(), output_bns().end(), obn) != output_bns().end())
        << "Cannot find output_arg_name : " << out_arg_name
        << " output_arg_index : " << out_arg_index << " in op_name: " << op_conf().name();

    std::string repeated_ibn_err_msg =
        "Cannot repeated set inplace proposal for same intput arg : " + in_arg_name
        + " index : " + std::to_string(in_arg_index) + " in op_name: " + op_conf().name();
    std::string repeated_obn_err_msg =
        "Cannot repeated set inplace proposal for same output arg : " + out_arg_name
        + " index : " + std::to_string(out_arg_index) + " in op_name: " + op_conf().name();
    CHECK_OR_RETURN(bn_in_op_unique_check.insert(ibn).second) << repeated_ibn_err_msg;
    CHECK_OR_RETURN(bn_in_op_unique_check.insert(obn).second) << repeated_obn_err_msg;
    return Maybe<void>::Ok();
  };
  JUST(kernel_reg_val->inplace_proposal_fn(infer_ctx, AddInplaceArgPairFn));
  return Maybe<void>::Ok();
}

LogicalBlobId UserOp::lbi4ibn(const std::string& input_bn) const {
  auto pair = GenUnRepeatedBn(input_bn);
  return GenLogicalBlobId(op_conf().user_conf().input().at(pair.first).s(pair.second));
}

LogicalBlobId UserOp::lbi4obn(const std::string& output_bn) const {
  auto pair = GenUnRepeatedBn(output_bn);
  auto ret = GenLogicalBlobId(op_conf().user_conf().output().at(pair.first).s(pair.second));
  CHECK_EQ(ret.op_name(), op_conf().name());
  CHECK_EQ(ret.blob_name(), output_bn);
  return ret;
}

Maybe<void> UserOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  if (val_->infer_sbp_signature_fn) {
    UserOpInferSbpSignatureFnContext ctx(this, sbp_signature, sbp_sig_conf, SbpInferHint4Ibn);
    return val_->infer_sbp_signature_fn(&ctx);
  } else {
    return Operator::InferSbpSignature(sbp_signature, sbp_sig_conf, CalcOrderValue4SbpSig,
                                       SbpInferHint4Ibn, parallel_desc);
  }
}

Maybe<void> UserOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const {
  CHECK_OR_RETURN(val_ != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in op registry!";
  UserOpSbpContext sbp_ctx(this, sbp_sig_list, LogicalBlobDesc4Ibn);
  JUST(val_->get_sbp_fn(&sbp_ctx));
  // Add Broadcast for source user op tick input
  std::string tick_bn = GenRepeatedBn(user_op::kUserSourceOpTickInputArgName, 0);
  if (val_->op_def.input_size() == 0 && input_bns().size() == 1) {
    CHECK_OR_RETURN(input_bns().Get(0) == tick_bn)
        << "user op_name: " << op_conf().name()
        << " op_type_name: " << op_conf().user_conf().op_type_name()
        << " set ERROR input arg name : " << input_bns().Get(0) << " because NO input in op def";
    for (auto& sbp_sig : *sbp_sig_list->mutable_sbp_signature()) {
      auto* bn2sbp = sbp_sig.mutable_bn_in_op2sbp_parallel();
      if (bn2sbp->find(tick_bn) == bn2sbp->end()) {
        (*bn2sbp)[tick_bn].mutable_broadcast_parallel();
      }
    }
  }
  // Check valid
  for (const auto& sbp_sig : sbp_sig_list->sbp_signature()) {
    const auto& bn2sbp = sbp_sig.bn_in_op2sbp_parallel();
    for (const auto& ibn : input_bns()) {
      auto pair = GenUnRepeatedBn(ibn);
      CHECK_OR_RETURN(bn2sbp.find(ibn) != bn2sbp.end())
          << "In op_name: " << op_conf().name()
          << " op_type_name: " << op_conf().user_conf().op_type_name()
          << ", input_arg_name : " << pair.first << " input_arg_index : " << pair.second
          << " have NOT set sbp signature";
    }
    for (const auto& obn : output_bns()) {
      auto pair = GenUnRepeatedBn(obn);
      CHECK_OR_RETURN(bn2sbp.find(obn) != bn2sbp.end())
          << "In op_name: " << op_conf().name()
          << " op_type_name: " << op_conf().user_conf().op_type_name()
          << ", output_arg_name : " << pair.first << " output_arg_index : " << pair.second
          << " have NOT set sbp signature";
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> UserOp::InferOpTimeShape(
    const std::function<Maybe<const Shape>(const std::string&)>& GetTimeShape4BnInOp,
    std::shared_ptr<const Shape>* time_shape) const {
  if (val_->infer_output_blob_time_shape_fn) {
    std::shared_ptr<Shape> op_time_shape(new Shape());
    UserOpInferOutputBlobTimeShapeFnContext infer_output_blob_time_shape_fn_ctx(
        this, GetTimeShape4BnInOp, op_time_shape.get());
    *time_shape = op_time_shape;
    return val_->infer_output_blob_time_shape_fn(&infer_output_blob_time_shape_fn_ctx);
  } else {
    return Operator::InferOpTimeShape(GetTimeShape4BnInOp, time_shape);
  }
}

Maybe<void> UserOp::InferParallelDistributionSignature(
    ParallelDistributionSignature* parallel_distribution_signature,
    const ParallelDistributionSignature& parallel_distribution_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
        ParallelDistributionInferHint4Ibn) const {
  if (val_->infer_parallel_distribution_fn) {
    UserOpInferParallelDistributionFnContext ctx(this, parallel_distribution_signature,
                                                 parallel_distribution_constraints,
                                                 ParallelDistributionInferHint4Ibn);
    return val_->infer_parallel_distribution_fn(&ctx);
  } else {
    return Operator::InferParallelDistributionSignature(
        parallel_distribution_signature, parallel_distribution_constraints, parallel_desc,
        ParallelDistributionInferHint4Ibn);
  }
}

Symbol<OperatorConf> UserOp::GetOpConfWithoutOpNameAndLbn() const {
  OperatorConf op_conf(this->op_conf());
  op_conf.set_name("undefined-op-name");
  UserOpConf* user_op_conf = op_conf.mutable_user_conf();
  for (auto& pair : *user_op_conf->mutable_input()) {
    for (auto& str : *pair.second.mutable_s()) { str = "undefined-op-name/undefined-ibn"; }
  }
  for (auto& pair : *user_op_conf->mutable_output()) {
    std::string prefix = "undefined-op-name/";
    prefix += pair.first;
    prefix += "_";
    int i = 0;
    for (auto& str : *pair.second.mutable_s()) { str = prefix + std::to_string(i++); }
  }
  return SymbolOf(op_conf);
}

void UserOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  auto user_conf = kernel_conf->mutable_user_conf();
  ForEachBnInOp([&](const std::string& bn) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);
    if (blob_desc) { blob_desc->ToProto(&(*user_conf->mutable_bn_in_op2blob_desc())[bn]); }
  });
}

const user_op::UserOpConfWrapper& UserOp::user_op_conf() const {
  CHECK(user_op_conf_);
  return *user_op_conf_;
}

REGISTER_OP(OperatorConf::kUserConf, UserOp);

}  // namespace oneflow
