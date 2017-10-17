#ifndef ONEFLOW_CORE_KERNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_H_

#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/operator/operator_manager.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel() = default;

  virtual void InitFromOpProto(const OperatorProto& op_proto);

  void InitModelBlobs(
      const KernelCtx& ctx, ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num, const Snapshot*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  virtual void InitModelTmpBlobs(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNEXPECTED_RUN();
  }

  // for Forward / Bp Calculation in FwExecGragh node and BpExecGragh node
  // through bn_in_op2blob_ptr function get the input blob and output blob
  // the Kernel will using the input blob calculate the result and fill output
  virtual void Forward(const KernelCtx& ctx,
                       std::function<Blob*(const std::string&)>) const = 0;
  virtual void Backward(const KernelCtx& ctx,
                        std::function<Blob*(const std::string&)>) const {
    UNEXPECTED_RUN();
  }

  //
  const std::string& Lbn4BnInOp(const std::string& bn_in_op) const {
    return op_->Lbn4BnInOp(bn_in_op);
  }

 protected:
  Kernel() = default;
  const Operator* op() const { return op_.get(); }

  virtual void InitModelBlobsWithDir(
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNEXPECTED_RUN();
  }
  virtual void InitModelBlobsWithRandomSeed(
      const KernelCtx& ctx, std::mt19937 random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNEXPECTED_RUN();
  }
  template<DeviceType device_type>
  void CopyDataIdToAllOb(DeviceCtx* ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob,
                         Blob* input_blob) const {
    for (const std::string& obn : op_->output_bns()) {
      Blob* output_blob = BnInOp2Blob(obn);
      output_blob->CopyDataIdFrom<device_type>(ctx, input_blob);
    }
  }
  template<DeviceType device_type>
  void CopyDataIdFromSoleIbToAllOb(
      DeviceCtx* ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    CopyDataIdToAllOb<device_type>(ctx, BnInOp2Blob,
                                   BnInOp2Blob(op_->SoleIbn()));
  }
  template<DeviceType device_type>
  void CopyDataIdFromSoleIbToAllObIfNeed(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    if (BnInOp2Blob(op_->SoleIbn())->has_data_id()) {
      CopyDataIdFromSoleIbToAllOb<device_type>(ctx.device_ctx, BnInOp2Blob);
    }
  }

 private:
  std::unique_ptr<const Operator> op_;
};

using KernelLaunchFunc = void (Kernel::*)(
    const KernelCtx&, std::function<Blob*(const std::string&)>) const;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_H_
