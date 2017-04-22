#ifndef ONEFLOW_OPERATOR_CONCAT_OP_H_
#define ONEFLOW_OPERATOR_CONCAT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ConcatOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatOp);
  ConcatOp() = default;
  ~ConcatOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;

  std::string normal_ibn2lbn(const std::string& input_bn) const override { TODO(); }
  std::string GetValueFromPbOpConf(const std::string& k) const override;
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  void InferShape4Mtb(ParallelPolicy, uint64_t parallel_id) const override {
    TODO();
  }
  void InferShape4Mdb(ParallelPolicy, uint64_t parallel_id) const override {
    TODO();
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_CONCAT_OP_H_
