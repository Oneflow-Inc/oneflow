#ifndef ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_
#define ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/logical_blob_parallel_desc.h"
#include "oneflow/core/operator/op_parallel_match_result.pb.h"
#include "oneflow/core/job/lbpd_hint.h"

namespace oneflow {

class OpParallelSignature final {
 public:
  using FnSbpInferHint4BnInOp = std::function<const SbpInferHint&(const std::string&)>;
  using FnGetMathResult = std::function<const OpParallelMatchResult(const FnSbpInferHint4BnInOp&,
                                                                    const ParallelContext*)>;
  using FnSignatureGeneratorFunc =
      std::function<void(const FnSbpInferHint4BnInOp&, HashMap<std::string, SbpParallel>*)>;
  OpParallelSignature(const std::string& description, const FnGetMathResult& get_match_result,
                      const FnSignatureGeneratorFunc& signature_generator)
      : description_(description),
        get_match_result_(get_match_result),
        signature_generator_(signature_generator) {}
  OpParallelSignature(const OpParallelSignature&) = default;
  ~OpParallelSignature() = default;

  const std::string Description() const { return description_; }
  const OpParallelMatchResult GetMatchResult(const FnSbpInferHint4BnInOp& SbpInferHint4BnInOp,
                                             const ParallelContext* parallel_ctx) const {
    return get_match_result_(SbpInferHint4BnInOp, parallel_ctx);
  }
  void GenerateSignature(const FnSbpInferHint4BnInOp& SbpInferHint4BnInOp,
                         HashMap<std::string, SbpParallel>* bn2sbp) const {
    signature_generator_(SbpInferHint4BnInOp, bn2sbp);
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
std::unique_ptr<const OpParallelSignature> MakeCloneOpParallelSignature(const Operator* op);

// (C, S(0), ...) -> (S(0), ...)
// return blobs: data splitted
// intput blobs: split data input blobs and clone model input blobs
std::unique_ptr<const OpParallelSignature> MakeOpParallelSignature_DS_MC_2_DS(const Operator* op);

// (C, S, ...) -> (S, ...)
// return blobs: model splitted
// input blobs: clone data input blobs and split model input blobs
std::unique_ptr<const OpParallelSignature> MakeOpParallelSignature_DC_MS_2_MS(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_
