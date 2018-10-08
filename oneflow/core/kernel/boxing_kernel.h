#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class BoxingKernel : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  BoxingKernel() = default;
  virtual ~BoxingKernel() = default;

 protected:
  virtual const BoxingOpConf& boxing_conf() const = 0;
  virtual const PbRpf<std::string>& InputBns() const = 0;
  virtual const PbRpf<std::string>& OutputBns() const = 0;
  const PbRpf<std::string>& ibn_0() const { return ibn_0_; }
  const PbRpf<std::string>& obn_0() const { return obn_0_; }

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  template<typename Iter>
  void ForwardField(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void ForwardDataId(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void ForwardColNum(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void ForwardVaryingInstanceNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ForwardInstanceVaryingElemCnt(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void SetColId(const KernelCtx&, std::function<Blob*(const std::string&)>) const;
  void SetMaxColId(const KernelCtx&, std::function<Blob*(const std::string&)>) const;

  PbRpf<std::string> ibn_0_;
  PbRpf<std::string> obn_0_;
};

struct BoxingKernelUtil {
  static void CopyFromFirstToOtherBlobs(DeviceCtx* ctx,
                                        std::function<Blob*(const std::string&)> BnInOp2Blob,
                                        const PbRpf<std::string>& bns, CopyBlobFieldMthd Copy);

  static PbRpf<std::string> ConstructPbRpf(const std::string& s);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
