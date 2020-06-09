#include "oneflow/core/common/util.h"
#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

namespace {

class DumpTimeShapeAndBlobParallelConfPass final : public OpGraphPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DumpTimeShapeAndBlobParallelConfPass);
  DumpTimeShapeAndBlobParallelConfPass() = default;
  ~DumpTimeShapeAndBlobParallelConfPass() override = default;

  bool IsEnabled() const override { return true; }

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const override {
    op_graph.DumpOpTimeShape(job);
    op_graph.DumpBatchAxisLbi(job);
    op_graph.DumpLogicalBlobDesc(job);
    op_graph.DumpSbpSignature(job);
    return Maybe<void>::Ok();
  }
};

REGISTER_FUNCTION_PASS("DumpTimeShapeAndBlobParallelConfPass",
                       DumpTimeShapeAndBlobParallelConfPass);

}  // namespace

}  // namespace oneflow
