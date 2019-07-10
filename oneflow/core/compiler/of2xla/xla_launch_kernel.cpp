#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/jit/xla_device_context.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"  // GetXLARandomSeed
#include "oneflow/core/compiler/of2xla/xla_allocator.h"
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

class CompilationContext {
 public:
  explicit CompilationContext(const std::string &builder_name,
                              DeviceType device_type,
                              int intra_op_num_threads) {
    namespace se = tensorflow::se;
    se::Platform::Id platform_id = nullptr;
    if (device_type == DeviceType::kCPU) {
      platform_id = se::host::kHostPlatformId;
    } else if (device_type == DeviceType::kGPU) {
      platform_id = se::cuda::kCudaPlatformId;
    }
    OF_CHECK_AND_ASSIGN(auto platform,
                        se::MultiPlatformManager::PlatformWithId(platform_id));
 
    // Get local client if the client of `device_type` has been created,
    // otherwise create a new local client
    xla::LocalClientOptions client_options;
    client_options.set_platform(platform);
    client_options.set_intra_op_parallelism_threads(intra_op_num_threads);
    OF_CHECK_AND_ASSIGN(
        client_, xla::ClientLibrary::GetOrCreateLocalClient(client_options));

    xla_allocator_ = std::make_shared<mola::XlaAllocator>(platform);

    builder_ = std::make_shared<xla::XlaBuilder>(absl::StrCat("XlaBuilder_",
                                                 builder_name));
    parallel_ctx_ = LocalParallelContext(device_type);
  }

  xla::LocalClient *client() const { return client_; }
  xla::XlaBuilder *builder() const { return builder_.get(); }

  mola::XlaAllocator *allocator() const { return xla_allocator_.get(); }  

  const ParallelContext &parallel_ctx() const { return parallel_ctx_; }
  
 private:
  xla::LocalClient *client_;
  std::shared_ptr<xla::XlaBuilder> builder_;

  std::shared_ptr<mola::XlaAllocator> xla_allocator_;

  ParallelContext parallel_ctx_;
};

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::BuildLocalExecutable(
    const CompilationContext &compile_ctx,
    const std::vector<Blob *> &entry_blobs,
    const std::vector<std::string> &entry_blob_names,
    mola::CompilationResult *compile_result) const {
  CHECK(this->op_conf().has_xla_launch_conf())
      << "BuildLocalExecutable need a `XlaLaunchOpConf`.";
  // Pass a fake local `ParallelContext` to the compiler in order to get
  // the shape of arguments by `InferBlobDescs`
  mola::XlaCompiler compiler(compile_ctx.client(), compile_ctx.builder(),
                             this->op_conf().xla_launch_conf(), device_type,
                             compile_ctx.parallel_ctx(), entry_blobs,
                             entry_blob_names, true /*force_compile*/);
  *compile_result = compiler.Compile();
}

template <DeviceType device_type, typename T>
void XlaLaunchKernel<device_type, T>::SyncRunExecutable(
    const CompilationContext &compile_ctx, xla::LocalExecutable *executable,
    const std::vector<Blob *> &entry_blobs,
    const std::vector<xla::Shape> &input_shapes,
    std::vector<Blob *> &output_blobs, const xla::Shape &output_shape) const {
  namespace se = tensorflow::se;
  CHECK_EQ(entry_blobs.size(), input_shapes.size())
      << "Mismatch between entry blobs size and input shapes size.";
  xla::LocalClient *client = compile_ctx.client();

  // Translate input blobs to xla ShapedBuffer suitable running the executable
  int argument_size = input_shapes.size();
  std::vector<std::shared_ptr<xla::ShapedBuffer>> shaped_buffers(argument_size);
  std::vector<xla::ShapedBuffer *> arguments(argument_size);
  for (int i = 0; i < input_shapes.size(); ++i) {
    const xla::Shape& shape = input_shapes[i];
    const xla::Shape on_device_shape =
        client->backend().transfer_manager()->HostShapeToDeviceShape(shape);
    CHECK(!on_device_shape.IsTuple()) << "Tuple shape is not allowed for input "
                                         "arguments in AsyncRunExecutable.";
    size_t data_size = entry_blobs[i]->ByteSizeOfDataContentField();
    const char *data_ptr = entry_blobs[i]->dptr<char>();
    se::DeviceMemoryBase memory_base =
        se::DeviceMemoryBase(const_cast<char *>(data_ptr), data_size);
    shaped_buffers[i] = std::make_shared<xla::ShapedBuffer>(
        /*on_host_shape=*/shape, /*on_device_shape=*/shape,
        client->platform(), client->default_device_ordinal());
    shaped_buffers[i]->set_buffer(memory_base, /*index=*/{});
    arguments[i] = shaped_buffers[i].get();
  }

  xla::ExecutableRunOptions run_options;
  run_options.set_stream(nullptr);
  run_options.set_allocator(compile_ctx.allocator());
  // run_options.set_intra_op_thread_pool();
  run_options.set_rng_seed(tensorflow::GetXLARandomSeed());
  OF_CHECK_AND_ASSIGN(auto run_result, executable->Run(arguments, run_options));
  
  // Translate result to output blobs
  if (!run_result.on_host_shape().IsTuple()) {
    xla::ShapedBuffer nontuple_buffer = run_result.release();
    xla::ShapedBuffer buffer(
        xla::ShapeUtil::MakeTupleShape({nontuple_buffer.on_host_shape()}),
        xla::ShapeUtil::MakeTupleShape({nontuple_buffer.on_device_shape()}),
        run_result.platform(), run_result.device_ordinal());
    buffer.buffers().CopySubtreeFrom(nontuple_buffer.buffers(),
                                     /*source_base_index=*/{},
                                     /*target_base_index=*/{0});
    run_result = xla::ScopedShapedBuffer(std::move(buffer),
                                         run_result.memory_allocator());
  }

//  for (int i = 0; i < output_blobs.size(); ++i) {
//    se::DeviceMemoryBase buffer = run_result.buffer({i});
//  }
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

  CompilationContext compile_ctx(this->op_conf().name(), device_type,
                                    1 /*intra_op_num_threads*/);
  mola::CompilationResult compile_result;
  BuildLocalExecutable(compile_ctx, entry_blobs, entry_blob_names,
                       &compile_result);

  xla::LocalExecutable *executable = compile_result.executable.get();
  CHECK(executable) << "Build executable failed.";
  SyncRunExecutable(compile_ctx, executable, entry_blobs,
                    compile_result.xla_input_shapes, output_blobs,
                    compile_result.xla_output_shape);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kXlaLaunchConf, XlaLaunchKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
