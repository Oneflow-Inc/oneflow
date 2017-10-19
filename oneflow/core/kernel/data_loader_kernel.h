#ifndef ONEFLOW_CORE_KERNEL_DATA_LOADER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DATA_LOADER_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/persistence/data_set_in_stream.h"

namespace oneflow {

template<typename T>
class DataLoaderKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoaderKernel);
  DataLoaderKernel() = default;
  ~DataLoaderKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
  void ForwardOld(const KernelCtx&,
                  std::function<Blob*(const std::string&)>) const;

  void ForwardNew(const KernelCtx&,
                  std::function<Blob*(const std::string&)>) const;

  void InitInStream(const KernelCtx&) const;
  void InitDataSetInStream(const KernelCtx&) const;

  mutable std::unique_ptr<PersistentInStream> in_stream_;
  mutable std::unique_ptr<DataSetInStream> data_set_in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_LOADER_KERNEL_H_
