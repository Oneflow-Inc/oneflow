#ifndef ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_
#define ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/operator/op_parallel_match_result.pb.h"
#include "oneflow/core/job/sbp_infer_hint.h"

namespace oneflow {

class Operator;

class OpParallelSignature {
 public:
  virtual ~OpParallelSignature() = default;
  virtual const std::string Description() const = 0;
  virtual const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelContext* parallel_ctx) const = 0;
  virtual void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const = 0;

 protected:
  OpParallelSignature(const Operator* op) : op_(op) {}
  const Operator& op() const { return *op_; }

 private:
  const Operator* op_;
};

class LambdaOpParallelSignature final : public OpParallelSignature {
 public:
  using FnSbpInferHint4Ibn = std::function<const SbpInferHint&(const std::string&)>;
  using FnGetMathResult =
      std::function<const OpParallelMatchResult(const FnSbpInferHint4Ibn&, const ParallelContext*)>;
  using FnSignatureGeneratorFunc =
      std::function<void(const FnSbpInferHint4Ibn&, HashMap<std::string, SbpParallel>*)>;
  LambdaOpParallelSignature(const std::string& description, const FnGetMathResult& get_match_result,
                            const FnSignatureGeneratorFunc& signature_generator)
      : OpParallelSignature(nullptr),
        description_(description),
        get_match_result_(get_match_result),
        signature_generator_(signature_generator) {}
  ~LambdaOpParallelSignature() override = default;

  const std::string Description() const override { return description_; }
  const OpParallelMatchResult GetMatchResult(const FnSbpInferHint4Ibn& SbpInferHint4Ibn,
                                             const ParallelContext* parallel_ctx) const override {
    return get_match_result_(SbpInferHint4Ibn, parallel_ctx);
  }
  void GenerateSignature(const FnSbpInferHint4Ibn& SbpInferHint4Ibn,
                         HashMap<std::string, SbpParallel>* bn2sbp) const override {
    signature_generator_(SbpInferHint4Ibn, bn2sbp);
  }

 private:
  const std::string description_;
  const FnGetMathResult get_match_result_;
  const FnSignatureGeneratorFunc signature_generator_;
};

const OpParallelMatchResult MakeOpParallelMatchSuccess();
const OpParallelMatchResult MakeOpParallelMatchSignatureMismatch();
const OpParallelMatchResult MakeOpParallelMatchParallelPolicyError(ParallelPolicy configured,
                                                                   ParallelPolicy expected);
const OpParallelMatchResult MakeOpParallelMatchParallelNumError(int64_t configured,
                                                                int64_t expected);

class Operator;

// (S(0), ...) -> (S(0), ...)
std::unique_ptr<const OpParallelSignature> MakeDataSplitOpParallelSignature(const Operator* op);

// (S,) -> (S, ...) or (C, ...) -> (S, ...)
std::unique_ptr<const OpParallelSignature> MakeModelSplitOpParallelSignature(const Operator* op);

// (C,) -> (C, ...)
std::unique_ptr<const OpParallelSignature> MakeModelBroadcastOpParallelSignature(
    const Operator* op);

// (C, S(0), ...) -> (S(0), ...)
// return blobs: data splitted
// intput blobs: split data input blobs and clone model input blobs
std::unique_ptr<const OpParallelSignature> Make_DS_MC_2_DS_OpParallelSignature(const Operator* op);

// (C, S, ...) -> (S, ...)
// return blobs: model splitted
// input blobs: clone data input blobs and split model input blobs
std::unique_ptr<const OpParallelSignature> Make_DC_MS_2_MS_OpParallelSignature(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_
