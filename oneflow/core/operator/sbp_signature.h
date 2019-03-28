#ifndef ONEFLOW_CORE_OPERATOR_SBP_SIGNATURE_H_
#define ONEFLOW_CORE_OPERATOR_SBP_SIGNATURE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/op_parallel_match_result.pb.h"
#include "oneflow/core/job/sbp_infer_hint.h"

namespace oneflow {

class Operator;

class SbpSignature {
 public:
  virtual ~SbpSignature() = default;
  virtual const std::string Description() const = 0;
  virtual const SbpSigMatchResult GetMatchResultIf(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const = 0;
  virtual void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const = 0;

 protected:
  SbpSignature(const Operator* op) : op_(op) {}
  const Operator& op() const { return *op_; }
  virtual const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const = 0;

 private:
  const Operator* op_;
};

class ParallelSbpSignature : public SbpSignature {
 protected:
  ParallelSbpSignature(const Operator* op) : SbpSignature(op) {}
  virtual ~ParallelSbpSignature() = default;
  virtual const SbpSigMatchResult GetMatchResultIf(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
};

const SbpSigMatchResult MakeSbpSigMatchSuccess();
const SbpSigMatchResult MakeSbpSigMatchSignatureMismatch();
const SbpSigMatchResult MakeSbpSigMatchParallelPolicyError(ParallelPolicy configured,
                                                           ParallelPolicy expected);
const SbpSigMatchResult MakeSbpSigMatchParallelNumError(int64_t configured, int64_t expected);
const SbpSigMatchResult MakeSbpSigMatchDeviceSetError(const std::string& configured,
                                                      const std::string& expected);

class Operator;

// (U, ...) -> (U, ...)
std::unique_ptr<const SbpSignature> MakeUnparallelSbpSignature(const Operator* op);

// (S(0), ...) -> (S(0), ...)
std::unique_ptr<const SbpSignature> MakeDataSplitSbpSignature(const Operator* op);

// (S,) -> (S, ...) or (B, ...) -> (S, ...)
std::unique_ptr<const SbpSignature> MakeModelSplitSbpSignature(const Operator* op);

// (B,) -> (B, ...)
std::unique_ptr<const SbpSignature> MakeBroadcastSbpSignature(const Operator* op);

// (B, S(0), ...) -> (S(0), ...)
// return blobs: data splitted
// intput blobs: split data input blobs and broadcast model input blobs
std::unique_ptr<const SbpSignature> Make_DS_MB_2_DS_SbpSignature(const Operator* op);

// (B, S, ...) -> (S, ...)
// return blobs: model splitted
// input blobs: broadcast data input blobs and split model input blobs
std::unique_ptr<const SbpSignature> Make_DB_MS_2_MS_SbpSignature(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SBP_SIGNATURE_H_
