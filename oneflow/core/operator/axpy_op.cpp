#include "oneflow/core/operator/axpy_op.h"

namespace oneflow {

namespace {

class AxpyOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AxpyOpParallelSignature);
  ~AxpyOpParallelSignature() override = default;

  AxpyOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (A, A) -> ()"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.parallel_num() != SbpInferHint4Ibn("y").parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
                                                 SbpInferHint4Ibn("y").parallel_num());
    }
    if (parallel_desc != SbpInferHint4Ibn("y").parallel_desc()) {
      return MakeOpParallelMatchDeviceSetError(
          parallel_desc.device_names(), SbpInferHint4Ibn("y").parallel_desc().device_names());
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["y"] = SbpInferHint4Ibn("y").sbp_parallel();
    (*bn2sbp)["x"] = SbpInferHint4Ibn("y").sbp_parallel();
  }
};

}  // namespace

void AxpyOp::InitFromOpConf() {
  EnrollInputBn("y");
  MutInputBlobModifier4Ibn("y")->set_is_mutable(true);
  EnrollInputBn("x", false);
}

void AxpyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  CHECK(*GetBlobDesc4BnInOp("x") == *GetBlobDesc4BnInOp("y"));
}

const PbMessage& AxpyOp::GetCustomizedConf() const { return op_conf().axpy_conf(); }

void AxpyOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new AxpyOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kAxpyConf, AxpyOp);

}  // namespace oneflow
