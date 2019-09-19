#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_KERNEL_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

#include "oneflow/engine/xla/of2xla/xla_compilation_cache.h"
#include "oneflow/engine/xla/of2xla/xla_launch_context.h"
#include "oneflow/engine/xla/of2xla/xla_graph_compiler.h"

namespace oneflow {

template <DeviceType device_type>
class XlaLaunchKernel : public KernelIf<device_type> {
 public:
  XlaLaunchKernel() = default;
  virtual ~XlaLaunchKernel() {}

 private:
  void ForwardDataContent(
      const KernelCtx &ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void BuildLocalExecutable(mola::XlaLaunchContext *launch_ctx,
                            const std::vector<Blob *> &entry_blobs,
                            const std::vector<std::string> &entry_blob_names,
                            const std::vector<std::string> &return_blob_names,
                            mola::CompilationResult **compile_result) const;

  void LaunchExecutable(mola::XlaLaunchContext *launch_ctx,
                        xla::LocalExecutable *executable,
                        const std::vector<Blob *> &entry_blobs,
                        const std::vector<xla::Shape> &input_shapes,
                        std::vector<Blob *> &output_blobs,
                        const xla::Shape &output_shape,
                        bool block_host_until_done) const;

  std::vector<Blob *> RealIOBuffers(
      const std::vector<Blob *> &input_blobs,
      const std::vector<Blob *> &output_blobs) const;

  mutable std::shared_ptr<mola::XlaCompilationCache> compilation_cache_;
};

}  // namespace oneflow

#endif  //ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_KERNEL_H_  

