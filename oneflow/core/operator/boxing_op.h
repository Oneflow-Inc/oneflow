#ifndef ONEFLOW_CORE_OPERATOR_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_BOXING_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BoxingOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingOp);
  BoxingOp() = default;
  ~BoxingOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 protected:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
  void InferDataTmpBlobDesc(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            std::vector<int64_t>* data_tmp_vec_ptr,
                            std::vector<int64_t>* data_tmp_dim0_inner_shape_vec_ptr) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BOXING_OP_H_
