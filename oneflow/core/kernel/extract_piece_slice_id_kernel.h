#ifndef ONEFLOW_CORE_OPERATOR_EXTRACT_PIECE_SLICE_ID_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_EXTRACT_PIECE_SLICE_ID_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class ExtractPieceSliceIdKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExtractPieceSliceIdKernel);
  ExtractPieceSliceIdKernel() = default;
  ~ExtractPieceSliceIdKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
struct ExtractPieceSliceIdUtil;

template<>
struct ExtractPieceSliceIdUtil<DeviceType::kCPU> {
  static void ForwardOneInOutPair(DeviceCtx* ctx, const int32_t instance_num,
                                  const int32_t slice_idx, int32_t* out_i_ptr);
};

template<>
struct ExtractPieceSliceIdUtil<DeviceType::kGPU> {
  static void ForwardOneInOutPair(DeviceCtx* ctx, const int32_t instance_num,
                                  const int32_t slice_idx, int32_t* out_i_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_EXTRACT_PIECE_SLICE_ID_KERNEL_OP_H_
