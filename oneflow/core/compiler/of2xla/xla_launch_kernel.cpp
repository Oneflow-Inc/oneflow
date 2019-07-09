#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_compiler.h"
#include "oneflow/core/compiler/of2xla/xla_launch_kernel.h"

namespace oneflow {

static ParallelContext LocalParallelContext(DeviceType device_type) {
  int device_id = 0;
  if (device_type == DeviceType::kGPU) {
#ifdef WITH_CUDA
    CudaCheck(cudaGetDevice(&device_id));
#endif
  }
  ParallelContext parallel_ctx;
  parallel_ctx.set_parallel_id(device_id);
  parallel_ctx.set_parallel_num(1);
  parallel_ctx.set_policy(kDataParallel);
  return parallel_ctx;
}

template <DeviceType device_type, typename T>
xla::LocalClient *XlaLaunchKernel<device_type, T>::GetOrCreateLocalClient(
    int intra_op_num_threads) const {
  namespace se = tensorflow::se;
  se::Platform::Id platform_id = nullptr;

  if (device_type == DeviceType::kCPU) {
    platform_id = se::host::kHostPlatformId;
  } else if (device_type == DeviceType::kGPU) {
    platform_id = se::cuda::kCudaPlatformId;
  }
  OF_CHECK_AND_ASSIGN(auto platform,
                      se::MultiPlatformManager::PlatformWithId(platform_id));

  xla::LocalClientOptions client_options;
  client_options.set_platform(platform);
  client_options.set_intra_op_parallelism_threads(intra_op_num_threads);
  OF_CHECK_AND_ASSIGN(
      auto client, xla::ClientLibrary::GetOrCreateLocalClient(client_options));
  return client;
}

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::BuildLocalExecutable(
    xla::LocalClient *client, const std::vector<Blob *> &entry_blobs,
    const std::vector<std::string> &entry_blob_names,
    mola::CompilationResult *compile_result) const {
  ParallelContext parallel_ctx = LocalParallelContext();

  // Pass a fake local `ParallelContext` to the compiler in order to get
  // the shape of arguments by `InferBlobDescs`
  mola::XlaCompiler compiler(client, this->op_conf(), device_type, parallel_ctx,
                             entry_blobs, entry_blob_names,
                             true /*force_compile*/);
  *compile_result = compiler.Compile();
}

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::AsyncRunExecutable(
    xla::LocalClient *client, xla::LocalExecutable *executable,
    const std::vector<Blob *> &entry_blobs,
    const std::vector<xla::Shape> &input_shapes,
    std::vector<Blob *> &output_blobs, const xla::Shape &output_shape) const {
  
  std::vector<xla::ShapedBuffer*> arguments;
  xla::ExecutableRunOptions run_options;
  executable->RunAsync(arguments, run_options);
}

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::ForwardDataContent(
                const KernelCtx &ctx,
                std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Prepare input and output blobs
  std::vector<Blob *> entry_blobs, output_blobs;
  std::vector<std::string> entry_blob_names;
  for (const auto& input_bn : this->op_attribute().input_bns()) {
    Blob* in_blob = BnInOp2Blob(input_bn);
    entry_blobs.push_back(in_blob);

    const LogicalBlobId& lbi = this->BnInOp2Lbi(input_bn);
    std::string blob_name = BlobName(lbi);
    entry_blob_names.push_back(blob_name);
  }

  for (const auto& output_bn : this->op_attribute().output_bns()) {
    Blob* in_blob = BnInOp2Blob(output_bn);
    output_blobs.push_back(in_blob);
  }

  // Get local client if the client of `device_type` has been created,
  // otherwise create a new local client and return it
  xla::LocalClient *client = GetOrCreateLocalClient(1 /*intra_op_num_threads*/);

  mola::CompilationResult compile_result;
  BuildLocalExecutable(client, entry_blobs, entry_blob_names, &compile_result);

  xla::LocalExecutable *executable = compile_result.executable.get();
  CHECK(executable) << "Executable build failed.";
  AsyncRunExecutable(client, executable, entry_blobs,
                     compile_result.xla_input_shapes, output_blobs,
                     compile_result.xla_output_shape);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kXlaLaunchConf, XlaLaunchKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
