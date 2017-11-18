#ifndef ONEFLOW_CORE_KERNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_H_

#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel() = default;

  void Init(const KernelConf&);

  void InitModelBlobs(
      const KernelCtx& ctx, const ParallelContext& parallel_ctx,
      const Snapshot*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  virtual void InitModelTmpBlobs(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  virtual void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  virtual void Backward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  const std::string& Lbn4BnInOp(const std::string& bn_in_op) const;

 protected:
  Kernel() = default;
  virtual void VirtualKernelInit() {}
  const KernelConf& kernel_conf() const { return kernel_conf_; }
  const OperatorConf& op_conf() const { return kernel_conf_.op_conf(); }

  virtual void InitModelBlobsWithRandomSeed(
      const KernelCtx& ctx, std::mt19937 random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  virtual void InitModelBlobsWithDir(
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  virtual void ForwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void ForwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void BackwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  virtual void BackwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

 private:
  KernelConf kernel_conf_;
};

template<DeviceType device_type>
class KernelIf : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelIf);
  virtual ~KernelIf() = default;

 protected:
  KernelIf() = default;

  virtual void ForwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void BackwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void CopyDataIdToAllOb(DeviceCtx* ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob,
                         const Blob* blob) const;
};

void AddKernelCreator(OperatorConf::OpTypeCase,
                      std::function<Kernel*(const KernelConf&)>);
void AddKernelCreator(OperatorConf::OpTypeCase, std::function<Kernel*()>);
std::unique_ptr<const Kernel> ConstructKernel(const KernelConf&);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_H_
