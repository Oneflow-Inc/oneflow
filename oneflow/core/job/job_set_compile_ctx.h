#ifndef ONEFLOW_CORE_JOB_SET_COMPILE_CTX_
#define ONEFLOW_CORE_JOB_SET_COMPILE_CTX_

#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_set_compile_ctx.pb.h"

namespace oneflow {

class JobSetCompileCtx final {
 public:
  JobSetCompileCtx() = default;
  ~JobSetCompileCtx() = default;

  PbMap<std::string, int64_t>* GetVarOpName2randomSeed() {
    return job_set_compile_ctx_proto_.mutable_var_op_name2random_seed();
  }

 private:
  JobSetCompileCtxProto job_set_compile_ctx_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SET_COMPILE_CTX_
