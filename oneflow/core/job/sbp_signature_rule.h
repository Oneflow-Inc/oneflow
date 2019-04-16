#ifndef ONEFLOW_CORE_JOB_SBP_SIGNATURE_RULE_H_
#define ONEFLOW_CORE_JOB_SBP_SIGNATURE_RULE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_sig_match_result.pb.h"
#include "oneflow/core/job/sbp_infer_hint.h"

namespace oneflow {

class Operator;

class SbpSignatureRule {
 public:
  virtual ~SbpSignatureRule() = default;
  virtual const std::string Description() const = 0;

  const SbpSigMatchResult MatchIf(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const SbpSignature& conf_obn_sbp_sig_hint, const ParallelDesc& parallel_desc) const;

  virtual void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const = 0;

 protected:
  SbpSignatureRule(const Operator* op) : op_(op) {}
  const Operator& op() const { return *op_; }

  virtual const SbpSigMatchResult MatchByParallelNum(int32_t parallel_num) const = 0;
  virtual const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const = 0;
  virtual const SbpSigMatchResult MatchByObnSbpSigHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const SbpSignature& conf_obn_sbp_signature) const = 0;

 private:
  const Operator* op_;
};

class ParallelSbpSignatureRule : public SbpSignatureRule {
 protected:
  ParallelSbpSignatureRule(const Operator* op) : SbpSignatureRule(op) {}
  virtual ~ParallelSbpSignatureRule() = default;
  const SbpSigMatchResult MatchByParallelNum(int32_t parallel_num) const override;
  const SbpSigMatchResult MatchByObnSbpSigHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const SbpSignature& conf_obn_sbp_signature) const override;
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
std::unique_ptr<const SbpSignatureRule> MakeUnparallelSbpSignatureRule(const Operator* op);

// (S(0), ...) -> (S(0), ...)
std::unique_ptr<const SbpSignatureRule> MakeDataSplitSbpSignatureRule(const Operator* op);

// (S,) -> (S, ...) or (B, ...) -> (S, ...)
std::unique_ptr<const SbpSignatureRule> MakeModelSplitSbpSignatureRule(const Operator* op);

// (B,) -> (B, ...)
std::unique_ptr<const SbpSignatureRule> MakeSoleIbnBroadcastSbpSignatureRule(const Operator* op);

// (B, ...) -> (B, ...)
std::unique_ptr<const SbpSignatureRule> MakeMultiIbnsBroadcastSbpSignatureRule(const Operator* op);

// (B, S(0), ...) -> (S(0), ...)
// return blobs: data splitted
// intput blobs: split data input blobs and broadcast model input blobs
std::unique_ptr<const SbpSignatureRule> Make_DS_MB_2_DS_SbpSignatureRule(const Operator* op);

// (B, S, ...) -> (S, ...)
// return blobs: model splitted
// input blobs: broadcast data input blobs and split model input blobs
std::unique_ptr<const SbpSignatureRule> Make_DB_MS_2_MS_SbpSignatureRule(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_SIGNATURE_RULE_H_
