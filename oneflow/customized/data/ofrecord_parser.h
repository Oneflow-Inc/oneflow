#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_PARSER_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_PARSER_H_

#include "oneflow/customized/data/parser.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

class OFRecordParser final : public Parser<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using BatchLoadTargetPtr = std::vector<LoadTargetPtr>;
  OFRecordParser() = default;
  ~OFRecordParser() = default;

  void Parse(std::shared_ptr<BatchLoadTargetPtr> batch_data,
             user_op::KernelComputeContext* ctx) override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_PARSER_H_
