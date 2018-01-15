#ifndef ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct DataLoadStatus {
  int32_t next_col_id;
  int32_t max_col_id;
  int64_t next_piece_id;
  bool is_eof;
};

template<typename T>
class BasicDataLoaderKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicDataLoaderKernel);
  BasicDataLoaderKernel() = default;
  ~BasicDataLoaderKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void ReadOnePieceToBlob(DataLoadStatus*, Blob*) const;
  void ReadOneColFromBufferToOutBlob(const KernelCtx&, DataLoadStatus*,
                                     const Blob* buffer_blob,
                                     Blob* out_blob) const;
  const char* ReadOneDataId(const char* line_ptr, Blob*, int64_t index) const;
  int32_t ReadOneDataContent(const char* line_ptr, Blob*, int64_t index) const;

  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BASIC_DATA_LOADER_KERNEL_H_
