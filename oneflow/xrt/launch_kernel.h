#ifndef ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_
#define ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/xrt/compilation_cache.h"
#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/types.h"

namespace oneflow {

template <DeviceType device_type>
class XrtLaunchKernel : public KernelIf<device_type> {
 public:
  XrtLaunchKernel() = default;
  virtual ~XrtLaunchKernel() {}

 private:
  void ForwardDataContent(
      const KernelCtx &ctx,
      std::function<Blob *(const std::string &)> BnInOp2Blob) const override;

  void MakeInputOutputAlias(const std::vector<xrt::Parameter> &entry_params,
                            std::vector<xrt::Parameter> *return_params,
                            std::vector<xrt::InputOutputAlias> *aliases) const;

  xrt::Executable *BuildExecutable(
      const std::vector<xrt::Parameter> &entry_params,
      const std::vector<xrt::Parameter> &return_params,
      const std::vector<xrt::InputOutputAlias> &aliases) const;

 private:
  mutable std::shared_ptr<xrt::CompilationCache> compilation_cache_;
};

}  // namespace oneflow

#endif  // ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_
