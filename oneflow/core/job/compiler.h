#ifndef ONEFLOW_CORE_JOB_COMPILER_H_
#define ONEFLOW_CORE_JOB_COMPILER_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class Compiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Compiler);
  Compiler() = default;
  ~Compiler() = default;

  Plan Compile();

 private:
  Plan DoCompile();
  void GenNetTopo(Plan* plan);
  void GenNcclTopo(Plan* plan);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPILER_H_
