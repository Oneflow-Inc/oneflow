#ifndef ONEFLOW_CORE_OPERATOR_NCCL_INTER_DEVICE_REDUCE_SUM_OP_H_
#define ONEFLOW_CORE_OPERATOR_NCCL_INTER_DEVICE_REDUCE_SUM_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NcclInterDeviceReduceSumOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclInterDeviceReduceSumOp)
  NcclInterDeviceReduceSumOp() = default;
  ~NcclInterDeviceReduceSumOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*, const OpContext*) const override;
  //  void InferOutBlobTimeShape(std::function<const Shape*(const std::string&)>
  //  GetTimeShape4BnInOp,
  //                             const ParallelContext* parallel_ctx, Shape* time_shape) const
  //                             override;

  void InferDiffBlobDescsWithoutFwBlob(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NCCL_INTER_DEVICE_REDUCE_SUM_OP_H_
