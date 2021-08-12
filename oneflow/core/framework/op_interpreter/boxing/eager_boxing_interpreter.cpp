#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"

namespace oneflow {

Maybe<EagerBoxingCall> EagerBoxingCall::New(Symbol<cfg::ParallelDistribution> in_nd_sbp,
                                            Symbol<cfg::ParallelDistribution> out_nd_sbp,
                                            Symbol<ParallelDesc> in_parallel_desc,
                                            Symbol<ParallelDesc> out_parallel_desc) {
  const auto* mgr = Global<EagerBoxingInterpreterManager>::Get();
  const auto& boxing_interpreter =
    JUST(mgr->GetEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
  return std::shared_ptr<EagerBoxingCall>(new EagerBoxingCall{
    .boxing_interpreter = boxing_interpreter,
    .in_nd_sbp = in_nd_sbp,
    .out_nd_sbp = out_nd_sbp,
    .in_parallel_desc = in_parallel_desc,
    .out_parallel_desc = out_parallel_desc,
  });
}

Maybe<one::Tensor> EagerBoxingCall::Apply(const std::shared_ptr<one::Tensor>& input) const {
  CHECK_OR_RETURN(JUST(input->nd_sbp()) == this->in_nd_sbp);
  CHECK_OR_RETURN(JUST(input->parallel_desc()) == this->in_parallel_desc);
  return this->boxing_interpreter->Interpret(
      input, this->in_nd_sbp, this->out_nd_sbp, this->in_parallel_desc, this->out_parallel_desc);
}

}
