#ifndef ONEFLOW_CUSTOMIZED_DATA_PARSER_H_
#define ONEFLOW_CUSTOMIZED_DATA_PARSER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

template<typename LoadTarget>
class Parser {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using BatchLoadTargetPtr = std::vector<LoadTargetPtr>;
  Parser() = default;
  virtual ~Parser() = default;

  virtual void Parse(std::shared_ptr<BatchLoadTargetPtr> batch_data,
                     user_op::KernelComputeContext* ctx) = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_PARSER_H_
