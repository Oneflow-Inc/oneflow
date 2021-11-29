#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

enum IRPassType : int32_t { kBeforeAD = 0, kAfterAD = 1 };

template<IRPassType ir_pass_type>
class IRRoundTrip final : public JobPass {
 public:
  IRRoundTrip() = default;
  ~IRRoundTrip() override = default;
  bool IsEnabled(const JobPassCtx& ctx) const;
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

}  // namespace oneflow
