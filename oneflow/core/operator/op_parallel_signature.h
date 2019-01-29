#ifndef ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_
#define ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_parallel_match_result.pb.h"

namespace oneflow {

struct OpParallelSignature final {
  using FnProducerLbParallelDesc4Ibn =
      std::function<const LogicalBlobParallelDesc&(const std::string&)>;
  using FnModelSplitAxis4BnInOp = std::function<int32_t(const std::string&)>;
  using FnGetMathResult = std::function<const OpParallelMatchResult(
      const FnProducerLbParallelDesc4Ibn&, const FnModelSplitAxis4BnInOp&, const ParallelContext*)>;
  using FnSignatureGeneratorFunc = std::function<void(
      const FnModelSplitAxis4BnInOp&, HashMap<std::string, LogicalBlobParallelDesc>*)>;
  OpParallelSignature(const std::string& desc, const FnGetMathResult& match_result,
                      const FnSignatureGeneratorFunc& gen_signature)
      : description(desc), get_match_result(match_result), signature_generator(gen_signature) {}
  OpParallelSignature(const OpParallelSignature&) = default;
  ~OpParallelSignature() = default;

  const std::string description;
  const FnGetMathResult get_match_result;
  const FnSignatureGeneratorFunc signature_generator;
};

const OpParallelMatchResult MakeOpParallelMatchSuccess();
const OpParallelMatchResult MakeOpParallelMatchSignatureMismatch();
const OpParallelMatchResult MakeOpParallelMatchParallelPolicyError(ParallelPolicy configured,
                                                                   ParallelPolicy expected);
const OpParallelMatchResult MakeOpParallelMatchParallelNumError(int64_t configured,
                                                                int64_t expected);

class Operator;
// (S(0), ...) -> (S(0), ...)
const OpParallelSignature MakeDataSplitOpParallelSignature(const Operator* op);
// (S,) -> (S, ...) or (C, ...) -> (S, ...)
const OpParallelSignature MakeModelSplitOpParallelSignature(const Operator* op);
// (C,) -> (C, ...)
const OpParallelSignature MakeCloneOpParallelSignature(const Operator* op);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OP_PARALLEL_SIGNATURE_H_
