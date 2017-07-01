#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class BoxingKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  BoxingKernel() = default;
  ~BoxingKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) override;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
  // Forward && backward will call these functions according to input-box types
  void AddCloneBoxForward(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)>) const;
  void AddCloneBoxBackward(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)>) const;
  void ConcatSplitBoxForward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)>) const;
  void ConcatSplitBoxBackward(const KernelCtx& ctx,
                              std::function<Blob*(const std::string&)>) const;
  void ConcatCloneBoxForward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)>) const;
  void ConcatCloneBoxBackward(const KernelCtx& ctx,
                              std::function<Blob*(const std::string&)>) const;
  void AddBoxes2Middle(const KernelCtx& ctx, const std::vector<std::string>&,
                       std::function<Blob*(const std::string&)>) const;
  void SplitMiddle2Boxes(const KernelCtx& ctx,
                         const std::vector<std::string>& bns,
                         std::function<Blob*(const std::string&)>,
                         bool reverse) const;
  void CopyMiddle2Boxes(const KernelCtx& ctx,
                        const std::vector<std::string>& bns,
                        std::function<Blob*(const std::string&)>) const;

  using WardFunc = void (BoxingKernel<device_type, FloatingPointType>::*)(
      const KernelCtx&, std::function<Blob*(const std::string&)>) const;

  WardFunc fw_func_;
  WardFunc bw_func_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
