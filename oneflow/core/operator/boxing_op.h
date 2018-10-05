#ifndef ONEFLOW_CORE_OPERATOR_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_BOXING_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BoxingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingOp);
  BoxingOp() = default;
  virtual ~BoxingOp() = default;

 protected:
  virtual const BoxingOpConf& boxing_conf() const = 0;
  virtual const PbRpf<std::string>& InputBns() const = 0;
  virtual const PbRpf<std::string>& OutputBns() const = 0;
  const PbMessage& GetCustomizedConf() const { return boxing_conf(); }
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
  std::vector<int64_t> CalcDataTmpBlobShapeVec(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      std::vector<int64_t>* instance_inner_shape_vec) const;
  void InferOutBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                         const std::vector<int64_t>& data_tmp_blob_shape_vec,
                         const std::vector<int64_t>& instance_inner_shape_vec) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BOXING_OP_H_
