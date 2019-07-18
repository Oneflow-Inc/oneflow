#include "oneflow/core/compiler/of2xla/xla_launch_kernel.h"

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"  // GetXLARandomSeed
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_stream.h"
#include "oneflow/core/compiler/of2xla/xla_compiler.h"
#include "oneflow/core/compiler/of2xla/xla_compilation_cache.h"
#include "oneflow/core/compiler/of2xla/xla_compilation_context.h"

namespace oneflow {

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::BuildLocalExecutable(
    const mola::CompilationContext &compile_ctx,
    const std::vector<Blob *> &entry_blobs,
    const std::vector<std::string> &entry_blob_names,
    const std::vector<std::string> &return_blob_names,
    mola::CompilationResult **compile_result) const {
  CHECK(this->op_conf().has_xla_launch_conf())
      << "BuildLocalExecutable need a `XlaLaunchOpConf`.";
  const auto &launch_conf = this->op_conf().xla_launch_conf();
  const auto &parallel_ctx = compile_ctx.parallel_ctx();

  if (!compilation_cache_) {
    compilation_cache_.reset(new mola::XlaCompilationCache);
  }

  const int device_ordinal = compile_ctx.device_ordinal();
  const mola::Signature signature = mola::ComputeSignature(
      compile_ctx.builder()->name(), device_ordinal, entry_blobs);
  bool force_compile = false;
  if (!force_compile) {
    *compile_result = compilation_cache_->GetRecord(signature);
  }

  if (!(*compile_result)) {
    mola::XlaLaunchGraph graph(launch_conf, device_type);

    // Pass a fake local `ParallelContext` to the compiler in order to get
    // the shape of arguments by `InferBlobDescs`
    mola::XlaCompiler compiler(compile_ctx.client(), compile_ctx.builder(),
                               &graph, parallel_ctx, entry_blobs,
                               entry_blob_names, return_blob_names);

    auto result = std::make_shared<mola::CompilationResult>();
    *result = compiler.Compile();
    // Record new compilation result
    compilation_cache_->Record(signature, result);
    // Get compilation result from cache
    *compile_result = compilation_cache_->GetRecord(signature);
  }
}

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::RunExecutable(
    const mola::CompilationContext &compile_ctx,
    xla::LocalExecutable *executable,
    const std::vector<Blob *> &entry_blobs,
    const std::vector<xla::Shape> &input_shapes,
    std::vector<Blob *> &output_blobs, const xla::Shape &output_shape) const {
  namespace se = tensorflow::se;
  CHECK_EQ(entry_blobs.size(), input_shapes.size())
      << "Size mismatch between entry blobs and input shapes.";
  const int device_ordinal = compile_ctx.device_ordinal();

  xla::LocalClient *client = compile_ctx.client();

  // Swap cuda stream between the backend stream and device context, so XLA
  // could launch kernel on the specified cuda stream of device context. Note
  // that it should do nothing for CPU mode in `SwapStreamHandle`
  OF_CHECK_AND_ASSIGN(auto stream,
                      client->mutable_backend()->BorrowStream(device_ordinal));
  mola::SwapStreamHandle<device_type>(stream.get(), compile_ctx.stream());

  // Translate input blobs to xla ShapedBuffer suitable running the executable
  int argument_size = input_shapes.size();
  std::vector<std::shared_ptr<xla::ShapedBuffer>> shaped_buffers(argument_size);
  std::vector<xla::ShapedBuffer *> arguments(argument_size);
  for (int i = 0; i < input_shapes.size(); ++i) {
    const xla::Shape& shape = input_shapes[i];
    const xla::Shape on_device_shape =
        client->backend().transfer_manager()->HostShapeToDeviceShape(shape);
    CHECK(!on_device_shape.IsTuple()) << "Tuple shape is not allowed for input "
                                         "arguments in RunExecutable.";
    size_t data_size = entry_blobs[i]->ByteSizeOfDataContentField();
    const char *data_ptr = entry_blobs[i]->dptr<char>();
    se::DeviceMemoryBase memory_base =
        se::DeviceMemoryBase(const_cast<char *>(data_ptr), data_size);
    shaped_buffers[i] = std::make_shared<xla::ShapedBuffer>(
        /*on_host_shape=*/shape, /*on_device_shape=*/shape,
        client->platform(), device_ordinal);
    shaped_buffers[i]->set_buffer(memory_base, /*index=*/{});
    arguments[i] = shaped_buffers[i].get();
  }

  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream.get());
  run_options.set_allocator(compile_ctx.allocator());
  run_options.set_intra_op_thread_pool(compile_ctx.host_device());
  run_options.set_rng_seed(tensorflow::GetXLARandomSeed());
  OF_CHECK_AND_ASSIGN(auto run_result, executable->Run(arguments, run_options));

  // Swap again to let the subsequent cuda kernels use the original stream
  mola::SwapStreamHandle<device_type>(stream.get(), compile_ctx.stream());
  // Result shape should be tuple
  CHECK(run_result.on_host_shape().IsTuple());

  // TODO(hjchen2) Reuse the allocated output blobs while runing the executable
  // Translate result to output blobs
  for (int i = 0; i < output_blobs.size(); ++i) {
    Blob *output = output_blobs[i];
    se::DeviceMemoryBase buffer = run_result.buffer({i});
    Memcpy<device_type>(compile_ctx.device_ctx(), output->mut_dptr(),
                        buffer.opaque(), output->ByteSizeOfDataContentField());
    // Maybe release result buffer
    // run_result.set_buffer(se::OwningDeviceMemory(), {i});
  }
}

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::ForwardDataContent(
                const KernelCtx &ctx,
                std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Prepare input and output buffers and their names
  std::vector<Blob *> entry_blobs, output_blobs;
  std::vector<std::string> entry_blob_names, return_blob_names;

  for (const auto& input_bn : this->op_attribute().input_bns()) {
    Blob* in_blob = BnInOp2Blob(input_bn);
    entry_blobs.push_back(in_blob);
    const LogicalBlobId& lbi = this->BnInOp2Lbi(input_bn);
    entry_blob_names.push_back(BlobName(lbi));
  }
  for (const auto& output_bn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(output_bn);
    output_blobs.push_back(out_blob);
    return_blob_names.push_back(output_bn);
  }
  
  mola::CompilationContext compile_ctx(this->op_conf().name(), ctx.device_ctx,
                                       device_type, 1 /*intra_op_num_threads*/);
  mola::CompilationResult *compile_result = nullptr;
  BuildLocalExecutable(compile_ctx, entry_blobs, entry_blob_names,
                       return_blob_names, &compile_result);

  CHECK(compile_result) << "Executable built failed.";
  auto *executable = compile_result->executable.get();

  // Run executable in synchronous mode for CPU, or asynchronous for GPU.
  RunExecutable(compile_ctx, executable, entry_blobs,
                compile_result->xla_input_shapes, output_blobs,
                compile_result->xla_output_shape);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kXlaLaunchConf, XlaLaunchKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
