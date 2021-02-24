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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

namespace {

void GenModelIoV2KernelConf(const VariableOpConf& variable_conf,
                            const ParallelContext& parallel_ctx, KernelConf* kernel_conf) {
  const Shape& logical_blob_shape = Shape(variable_conf.shape());
  SbpParallel sbp_parallel;
  if (variable_conf.split_axis().has_value()) {
    sbp_parallel.mutable_split_parallel()->set_axis(variable_conf.split_axis().value());
  } else {
    sbp_parallel.mutable_broadcast_parallel();
  }
  BlobDesc blob_desc(variable_conf.data_type());
  blob_desc.mut_shape() = Shape(logical_blob_shape);
  const std::vector<TensorSliceView> slices = SubTskGphBuilderUtil::GetTensorSliceView(
      parallel_ctx.parallel_num(), sbp_parallel, blob_desc);
  for (const auto& slice : slices) {
    slice.ToProto(kernel_conf->mutable_model_io_v2_conf()->mutable_slice_view()->Add());
  }
  *kernel_conf->mutable_model_io_v2_conf()->mutable_parallel_ctx() = parallel_ctx;
}

}  // namespace

class ModelInitV2Op : public Operator {
 public:
  void InitFromOpConf() override {
    CHECK(op_conf().has_model_init_v2_conf());
    EnrollInputBn("ref", false)->set_is_mutable(true);
    EnrollOutputBn("out", false);
    EnrollInputBn("tick", false);
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({1});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["ref"] =
        JUST(SbpInferHint4Ibn("ref"))->sbp_parallel();
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["out"].mutable_split_parallel()->set_axis(0);
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["tick"].mutable_broadcast_parallel();
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    GenModelIoV2KernelConf(op_conf().model_init_v2_conf().original_variable_conf(), *parallel_ctx,
                           kernel_conf);
  }
};

REGISTER_OP(OperatorConf::kModelInitV2Conf, ModelInitV2Op);

class ModelLoadV2Op : public Operator {
 public:
  void InitFromOpConf() override {
    CHECK(op_conf().has_model_load_v2_conf());
    EnrollInputBn("path", false);
    EnrollInputBn("ref", false)->set_is_mutable(true);
    EnrollOutputBn("out", false);
    EnrollInputBn("tick", false);
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({1});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["ref"] =
        JUST(SbpInferHint4Ibn("ref"))->sbp_parallel();
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["tick"].mutable_broadcast_parallel();
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["path"].mutable_broadcast_parallel();
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["out"].mutable_split_parallel()->set_axis(0);
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    GenModelIoV2KernelConf(op_conf().model_load_v2_conf().original_variable_conf(), *parallel_ctx,
                           kernel_conf);
  }
};

REGISTER_OP(OperatorConf::kModelLoadV2Conf, ModelLoadV2Op);

class ModelSaveV2Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveV2Op);
  ModelSaveV2Op() = default;
  ~ModelSaveV2Op() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_model_save_v2_conf());
    EnrollInputBn("path", false);
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
    EnrollInputBn("tick", false);
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({1});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["in"] =
        JUST(SbpInferHint4Ibn("in"))->sbp_parallel();
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["tick"].mutable_broadcast_parallel();
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["path"].mutable_broadcast_parallel();
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())["out"].mutable_split_parallel()->set_axis(0);
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    GenModelIoV2KernelConf(op_conf().model_save_v2_conf().original_variable_conf(), *parallel_ctx,
                           kernel_conf);
  }
};

REGISTER_OP(OperatorConf::kModelSaveV2Conf, ModelSaveV2Op);

}  // namespace oneflow
