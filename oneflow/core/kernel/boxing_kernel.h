#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include <string>
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {
  
template<DeviceType device_type, typename floating_point_type>
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
  // For concat ==> (split/clone) box:
  // a copy_rule means a step of memory action during runtime. Since the 
  // blob shapes are fixed after initilization, the offsets of each blobs 
  // are therefore constant during execution. We can write down the offsets
  // of each blob in the first time, and directly use these records to do
  // memory-copy in the next running.
  struct copy_rule {
    std::string src_bn, dst_bn; /* Blob names*/
    uint64_t src_offset, dst_offset; /* corresponding offsets*/
    uint64_t copy_sz; 
  };

  // Infer copy rules from BnInOp2BlobPtr
  void InferCopyRules(std::function<Blob*(const std::string&)>) const;

  void ConstructCopyRulesFromSlice(
      const std::map<const std::string*, int64_t>& src_bn2slice, 
      const std::map<const std::string*, int64_t>& dst_bn2slice,
      int64_t seg_cnt, int64_t slice_sz, int32_t concat_axis, 
      std::vector<struct copy_rule>* rules) const; 

  // Construct rules of copying first output blob to the remaining blobs
  void ConstructFwCloneRules(std::function<Blob*(const std::string&)>) const;

  // Infer copy rules with the assigned src && dst blob names
  void InferCopyRulesFromBns(
      std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
      const std::vector<std::string>& src_bns,
      const std::vector<std::string>& dst_bns, 
      std::vector<copy_rule>* copy_rules) const;

  // Do direct memory copy from saved rules
  void CopyDataFromRules( const KernelCtx& ctx, 
      std::function<Blob*(const std::string&)> BnInOp2BlobPtr, 
      const std::vector<copy_rule>& copy_rules) const; 

  // Forward function for Add box
  void AddBoxForward(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)>) const;
  // Backward function for Add box
  void AddBoxBackward(const KernelCtx& ctx,
                      std::function<Blob*(const std::string&)>) const;

  // Forward function for Concat box 
  void ConcatBoxForward(const KernelCtx& ctx,
                        std::function<Blob*(const std::string&)>) const;

  // Backward function for Concat-split box && concat-clone box
  void ConcatBoxBackward(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)>) const;

  using ExecFunc = void (BoxingKernel<device_type, floating_point_type>::*) 
    (const KernelCtx&, std::function<Blob*(const std::string&)>) const;

  mutable std::vector<copy_rule> fw_copy_rules;
  mutable std::vector<copy_rule> bw_copy_rules;
  ExecFunc fw_func_;
  ExecFunc bw_func_;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
