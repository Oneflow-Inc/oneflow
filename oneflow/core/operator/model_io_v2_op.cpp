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
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().model_init_v2_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("ref", 0)
        .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("ref"))->shape().NumAxes())
        .Build(sbp_sig_list);
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
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().model_load_v2_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    FOR_RANGE(int64_t, i, 0, JUST(LogicalBlobDesc4Ibn("ref"))->shape().NumAxes()) {
      SbpSignatureBuilder().Broadcast("path").Split("ref", i).Build(
          sbp_sig_list->mutable_sbp_signature()->Add());
    }
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
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().model_save_v2_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->set_data_type(DataType::kFloat);
    out->mut_shape() = Shape({1});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    BatchAxis4BnInOp("out")->clear_value();
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    FOR_RANGE(int64_t, i, 0, JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes()) {
      SbpSignatureBuilder().Broadcast("path").Split("in", i).Split("out", 0).Build(
          sbp_sig_list->mutable_sbp_signature()->Add());
    }
    return Maybe<void>::Ok();
  };

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    GenModelIoV2KernelConf(op_conf().model_save_v2_conf().original_variable_conf(), *parallel_ctx,
                           kernel_conf);
  }
};

REGISTER_OP(OperatorConf::kModelSaveV2Conf, ModelSaveV2Op);

}  // namespace oneflow
