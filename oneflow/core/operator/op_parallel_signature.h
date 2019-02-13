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

const OpParallelMatchResult MakeOpParallelMatchSuccess();
const OpParallelMatchResult MakeOpParallelMatchSignatureMismatch();
const OpParallelMatchResult MakeOpParallelMatchParallelPolicyError(ParallelPolicy configured,
                                                                   ParallelPolicy expected);
const OpParallelMatchResult MakeOpParallelMatchParallelNumError(int64_t configured,
                                                                int64_t expected);

class Operator;

// (S(0), ...) -> (S(0), ...)
std::unique_ptr<const OpParallelSignature> MakeDataSplitOpParallelSignature(const Operator* op);

// (S,) -> (S, ...) or (B, ...) -> (S, ...)
std::unique_ptr<const OpParallelSignature> MakeModelSplitOpParallelSignature(const Operator* op);

// (B,) -> (B, ...)
std::unique_ptr<const OpParallelSignature> MakeModelBroadcastOpParallelSignature(
    const Operator* op);

// (B, S(0), ...) -> (S(0), ...)
// return blobs: data splitted
// intput blobs: split data input blobs and broadcast model input blobs
std::unique_ptr<const OpParallelSignature> Make_DS_MB_2_DS_OpParallelSignature(const Operator* op);

// (B, S, ...) -> (S, ...)
// return blobs: model splitted
// input blobs: broadcast data input blobs and split model input blobs
std::unique_ptr<const OpParallelSignature> Make_DB_MS_2_MS_OpParallelSignature(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_
