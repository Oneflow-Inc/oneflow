#include "oneflow/core/kernel/concat_gt_proposals_kernel.h"
// #include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

template<typename T>
void ConcatGtProposalsKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void ConcatGtProposalsKernel<T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void ConcatGtProposalsKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void ConcatGtProposalsKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kConcatGtProposalsConf, ConcatGtProposalsKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
