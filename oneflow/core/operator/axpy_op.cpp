#include "oneflow/core/operator/axpy_op.h"

namespace oneflow {

namespace {

class AxpySbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AxpySbpSignature);
  ~AxpySbpSignature() override = default;

  AxpySbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (A, A) -> ()"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.parallel_num() != SbpInferHint4Ibn("y").parallel_num()) {
      return MakeSbpSigMatchParallelNumError(parallel_desc.parallel_num(),
                                             SbpInferHint4Ibn("y").parallel_num());
    }
    if (parallel_desc != SbpInferHint4Ibn("y").parallel_desc()) {
      return MakeSbpSigMatchDeviceSetError(parallel_desc.device_names(),
                                           SbpInferHint4Ibn("y").parallel_desc().device_names());
    }
    return MakeSbpSigMatchSuccess();
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
  EnrollInputBn("y")->set_is_mutable(true);
  EnrollInputBn("x", false);
}

void AxpyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  CHECK(*GetBlobDesc4BnInOp("x") == *GetBlobDesc4BnInOp("y"));
}

const PbMessage& AxpyOp::GetCustomizedConf() const { return op_conf().axpy_conf(); }

void AxpyOp::GetSbpSignatures(
    std::vector<std::unique_ptr<const SbpSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new AxpySbpSignature(this));
}

REGISTER_OP(OperatorConf::kAxpyConf, AxpyOp);

}  // namespace oneflow
