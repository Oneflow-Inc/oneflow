#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_KERNEL_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

#include "oneflow/core/compiler/of2xla/xla_compilation_cache.h"
#include "oneflow/core/compiler/of2xla/xla_compilation_context.h"
#include "oneflow/core/compiler/of2xla/xla_compiler.h"

namespace oneflow {

template <DeviceType device_type, typename T>
class XlaLaunchKernel : public KernelIf<device_type> {
 public:
  XlaLaunchKernel() = default;
  virtual ~XlaLaunchKernel() {}

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  void BuildLocalExecutable(const mola::CompilationContext &launch_ctx,
                            const std::vector<Blob *> &entry_blobs,
                            const std::vector<std::string> &entry_blob_names,
                            const std::vector<std::string> &return_blob_names,
                            mola::CompilationResult **compile_result) const;

  void SyncRunExecutable(const mola::CompilationContext &launch_ctx,
                         xla::LocalExecutable *executable,
                         const std::vector<Blob *> &entry_blobs,
                         const std::vector<xla::Shape> &input_shapes,
                         std::vector<Blob *> &output_blobs,
                         const xla::Shape &output_shape) const;

  mutable std::shared_ptr<mola::XlaCompilationCache> compilation_cache_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_KERNEL_H_

