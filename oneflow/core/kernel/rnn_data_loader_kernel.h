#ifndef ONEFLOW_CORE_KERNEL_RNN_DATA_LOADER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RNN_DATA_LOADER_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<typename T>  // T must be integer, like int32_t, int64_t
class RnnDataLoaderKernel final : public Kernel {
public:
    OF_DISALLOW_COPY_AND_MOVE(RnnDataLoaderKernel)
    RnnDataLoaderKernel() = default;
    ~RnnDataLoaderKernel() = default;

    void Forward(const KernelCtx&,
            std::function<Blob*(const std::string&)>) const override;

private:
    void InitInStream(int64_t) const;

    mutable std::unique_ptr<PersistentInStream> in_stream_;
};

}

#endif //ONEFLOW_CORE_KERNEL_RNN_DATA_LOADER_KERNEL_H_
