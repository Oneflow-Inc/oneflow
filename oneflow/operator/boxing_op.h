#ifndef ONEFLOW_GRAPH_BOXING_OP_H_
#define ONEFLOW_GRAPH_BOXING_OP_H_

#include "operator/operator.h"

namespace oneflow {

class BoxingOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingOp);
  BoxingOp() = default;
  ~BoxingOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }
  
  std::string ibn2lbn(const std::string& input_blob_name) const override;
  std::string obn2lbn(const std::string& output_blob_name) const override;
  std::string idbn2lbn(const std::string& input_diff_blob_name) const override;
  std::string odbn2lbn(const std::string& output_diff_blob_name) const override;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BOXING_OP_H_
