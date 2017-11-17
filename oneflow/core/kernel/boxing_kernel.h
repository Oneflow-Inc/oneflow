#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<typename T>
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
  // A CopyRule means a step of memory action during runtime. Since the
  // blob shapes are fixed after initilization, the offsets of each blobs
  // are therefore constant during execution. We can write down the offsets
  // of each blob in the first time, and directly use these records to do
  // memory-copy in the next running.
  struct CopyRule {
    bool is_data_id;
    std::string src_bn;
    std::string dst_bn;
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t copy_sz;
  };

  void InferFwCopyRules(std::function<Blob*(const std::string&)>) const;
  void InferBwCopyRules(std::function<Blob*(const std::string&)>) const;

  void InferCopyRulesFromConcatDim(
      const std::map<const std::string*, int64_t>& src_bn2concat_dim,
      const std::map<const std::string*, int64_t>& dst_bn2concat_dim,
      int64_t seg_cnt, int64_t concat_dim_sz, int32_t concat_axis,
      std::vector<CopyRule>* rules) const;

  // Infer rules of copying first output blob to the remaining output blobs
  void InferFwCloneRules(std::function<Blob*(const std::string&)>) const;

  // Infer copy rules with the assigned src && dst blob names
  void InferCopyRulesFromBns(
      std::function<Blob*(const std::string&)> BnInOp2Blob,
      const std::vector<std::string>& src_bns,
      const std::vector<std::string>& dst_bns, const int32_t src_axis,
      const int32_t dst_axis, std::vector<CopyRule>* copy_rules) const;

  // Infer copy rules in case that src && dst blobs are at same concat/split
  // axis
  void InferCopyRulesFromEqualAxis(
      std::function<Blob*(const std::string&)> BnInOp2Blob,
      const std::vector<std::string>& src_bns,
      const std::vector<std::string>& dst_bns,
      std::vector<CopyRule>* copy_rules) const;

  // Infer copy rules in case that src && dst blobs do not have equal axis
  void InferCopyRulesFromUnequalAxis(const std::vector<std::string>& src_bns,
                                     const std::vector<std::string>& dst_bns,
                                     const std::vector<Blob*> src_blobs,
                                     const std::vector<Blob*> dst_blobs,
                                     const int32_t src_concat_axis,
                                     const int32_t dst_split_axis,
                                     std::vector<CopyRule>* copy_rules) const;
  // Infer copy rules for data id
  void InferDataIdCopyRules(const std::vector<std::string>& src_bns,
                            const std::vector<std::string>& dst_bns,
                            const std::vector<Blob*> src_blobs,
                            const std::vector<Blob*> dst_blobs,
                            const int32_t src_concat_axis,
                            const int32_t dst_split_axis,
                            std::vector<CopyRule>* rules) const;

  // Do direct memory copy from saved rules
  void CopyDataFromRules(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob,
                         const std::vector<CopyRule>& copy_rules) const;

  // Forward && backward will call these functions according to input-box types
  void AddBoxForward(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)>) const;
  void AddBoxBackward(const KernelCtx& ctx,
                      std::function<Blob*(const std::string&)>) const;

  void ConcatBoxForward(const KernelCtx& ctx,
                        std::function<Blob*(const std::string&)>) const;

  void ConcatBoxBackward(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)>) const;

  using WardFunc = void (BoxingKernel<T>::*)(
      const KernelCtx&, std::function<Blob*(const std::string&)>) const;

  // NOTE: Due to current design, there is only one boxing thread, thus no
  // mutex is required here.
  mutable std::vector<CopyRule> fw_copy_rules_;
  mutable std::vector<CopyRule> bw_copy_rules_;
  WardFunc fw_func_;
  WardFunc bw_func_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
