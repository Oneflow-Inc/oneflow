#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include <string>
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {
  
template<DeviceType device_type, typename floating_point_type>
class BoxingKernel final {
};

template<typename floating_point_type>
class BoxingKernel<device_type, floating_point_type> 
    : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  BoxingKernel() = default;
  ~BoxingKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto);

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
 private:
  /* For concat ==> (split/clone) box:
   * a copy_rule means a step of memory action during runtime. Since the 
   * blob shapes are fixed after initilization, the offsets of each blobs 
   * are therefore constant during execution. We can write down the offsets
   * of each blob in the first time, and directly use these records to do
   * memory-copy in the next running.
   */
  struct copy_rule {
    string src_bn, dst_bn; /* Blob names*/
    uint64_t src_offset, dst_offset; /* corresponding offsets*/
    uint64_t copy_sz; /* memory size*/
  };
  void InferCopyRules(std::function<Blob*(const std::string&)>);
  void ConstructRulesFromShape(
      std::vector<const string&>& bns,
      std::vector<const Shape&>& in_shapes,
      std::vector<const Shape&>& out_shapes,
      uint64_t block_sz, uint64_t block_sz_0,
      std::vector<struct copy_rules>& ccis);

  void InferCopyRulesFromBns(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
    std::vector<string>& src_bns,
    std::vector<string>& dst_bns,
    std::vector<copy_rule>& copy_rules);

  void AddBoxForward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;
  void AddBoxBackward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;
  void ConcatBoxForward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;
  void ConcatBoxBackward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;

  using ExecFunc = void (BoxingKernel::*) (
      const KernelCtx&, 
      std::function<Blob*(const std::string&)>) const;

  std::vector<copy_action_info> fw_copy_rules;
  std::vector<copy_info> bw_copy_rules;
  ExecFunc fw_func_;
  ExecFunc bw_func_;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
