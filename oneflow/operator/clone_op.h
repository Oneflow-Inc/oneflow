#ifndef ONEFLOW_OPERATOR_CLONE_OP_H_
#define ONEFLOW_OPERATOR_CLONE_OP_H_

#include "operator/operator.h"

namespace oneflow {

class CloneOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneOp);
  CloneOp() = default;
  ~CloneOp() = default;

  void Init(const OperatorConf& op_conf) override;

  std::string ibn2lbn(const std::string& input_blob_name) const override;
  std::string obn2lbn(const std::string& output_blob_name) const override;
  std::string idbn2lbn(const std::string& input_diff_blob_name) const override;
  std::string odbn2lbn(const std::string& output_diff_blob_name) const override;

  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_CLONE_OP_H_
