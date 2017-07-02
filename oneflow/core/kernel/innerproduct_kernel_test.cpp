#include "oneflow/core/kernel/innerproduct_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

namespace {

enum class Location { kHost, kDevice };

template<typename FloatingPointType>
Blob* CreateBlob(const std::vector<int64_t>& dim_vec, FloatingPointType* matrix,
                 Location mem_location) {
  void* dptr;
  Shape* shape = new Shape(dim_vec);

  size_t dptr_size = shape->elem_cnt() * sizeof(FloatingPointType);
  if (mem_location == Location::kHost) {
    CHECK_EQ(cudaMallocHost(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, matrix, dptr_size, cudaMemcpyHostToHost),
             cudaSuccess);
  } else {
    CHECK_EQ(cudaMalloc(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, matrix, dptr_size, cudaMemcpyHostToDevice),
             cudaSuccess);
  }

  return new Blob(dptr, shape);
}

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
    bool has_bias_term) {
  Location loc;
  if (device_type == DeviceType::kCPU) {
    loc = Location::kHost;
  } else {
    loc = Location::kDevice;
  }

  // Create matrix
  FloatingPointType in_mat[] = {1, 2, 3, 4, 5, 6, 7, 8};
  FloatingPointType weight_mat[] = {5, 4, 5, 3, 2, 1, 7, 0, 1, 1, 9, 8};
  FloatingPointType bias_mat[] = {2, 3, 5};
  FloatingPointType bias_multiplier_mat[] = {1, 1};
  FloatingPointType out_mat[6] = {0};
  FloatingPointType in_diff_mat[8] = {0};
  FloatingPointType weight_diff_mat[12] = {0};
  FloatingPointType bias_diff_mat[3] = {0};

  FloatingPointType expected_out_without_bias_mat[] = {40,  25, 62,
                                                       108, 65, 138};
  FloatingPointType expected_in_diff_without_bias_mat[] = {
      312, 247, 933, 616, 808, 635, 2237, 1428};
  FloatingPointType expected_weight_diff_without_bias_mat[] = {
      580, 728, 876, 1024, 350, 440, 530, 620, 752, 952, 1152, 1352};

  FloatingPointType expected_out_mat[] = {42, 28, 67, 110, 68, 143};
  FloatingPointType expected_in_diff_mat[] = {333, 263, 1009, 662,
                                              829, 651, 2313, 1474};
  FloatingPointType expected_weight_diff_mat[] = {
      592, 744, 896, 1048, 368, 464, 560, 656, 782, 992, 1202, 1412};
  FloatingPointType expected_bias_diff_mat[] = {152, 96, 210};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  // Build blob for test
  (*bn2blob_ptr)["in"] = CreateBlob<FloatingPointType>({2, 4}, in_mat, loc);
  (*bn2blob_ptr)["weight"] =
      CreateBlob<FloatingPointType>({3, 4}, weight_mat, loc);

  (*bn2blob_ptr)["out"] = CreateBlob<FloatingPointType>({2, 3}, out_mat, loc);
  (*bn2blob_ptr)["out_diff"] = (*bn2blob_ptr)["out"];
  (*bn2blob_ptr)["in_diff"] =
      CreateBlob<FloatingPointType>({2, 4}, in_diff_mat, loc);
  (*bn2blob_ptr)["weight_diff"] =
      CreateBlob<FloatingPointType>({3, 4}, weight_diff_mat, loc);

  if (has_bias_term) {
    (*bn2blob_ptr)["bias"] =
        CreateBlob<FloatingPointType>({1, 3}, bias_mat, loc);
    (*bn2blob_ptr)["bias_multiplier"] =
        CreateBlob<FloatingPointType>({2, 1}, bias_multiplier_mat, loc);
    (*bn2blob_ptr)["bias_diff"] =
        CreateBlob<FloatingPointType>({1, 3}, bias_diff_mat, loc);
    (*bn2blob_ptr)["expected_bias_diff"] =
        CreateBlob<FloatingPointType>({1, 3}, expected_bias_diff_mat, loc);
    (*bn2blob_ptr)["expected_out"] =
        CreateBlob<FloatingPointType>({2, 3}, expected_out_mat, loc);
    (*bn2blob_ptr)["expected_in_diff"] =
        CreateBlob<FloatingPointType>({2, 4}, expected_in_diff_mat, loc);
    (*bn2blob_ptr)["expected_weight_diff"] =
        CreateBlob<FloatingPointType>({3, 4}, expected_weight_diff_mat, loc);
  } else {
    (*bn2blob_ptr)["expected_out"] = CreateBlob<FloatingPointType>(
        {2, 3}, expected_out_without_bias_mat, loc);
    (*bn2blob_ptr)["expected_in_diff"] = CreateBlob<FloatingPointType>(
        {2, 4}, expected_in_diff_without_bias_mat, loc);
    (*bn2blob_ptr)["expected_weight_diff"] = CreateBlob<FloatingPointType>(
        {3, 4}, expected_weight_diff_without_bias_mat, loc);
  }
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type>
void BuildKernelCtx(KernelCtx* ctx);

template<>
void BuildKernelCtx<DeviceType::kCPU>(KernelCtx* ctx) {
  auto cpu_stream = new CpuStream;
  ctx->device_ctx = new CpuDeviceCtx(cpu_stream);
}

template<>
void BuildKernelCtx<DeviceType::kGPU>(KernelCtx* ctx) {
  cudaStream_t* cuda_stream = new cudaStream_t;
  cublasHandle_t* cublas_handle = new cublasHandle_t;
  CHECK_EQ(cudaStreamCreate(cuda_stream), cudaSuccess);
  CHECK_EQ(cublasCreate(cublas_handle), CUBLAS_STATUS_SUCCESS);
  CHECK_EQ(cublasSetStream(*cublas_handle, *cuda_stream),
           CUBLAS_STATUS_SUCCESS);
  ctx->device_ctx = new CudaDeviceCtx(cuda_stream, cublas_handle, nullptr);
}

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildInnerProductKernel(bool has_bias_term) {
  // Config InnerProduct operator
  OperatorConf op_conf;
  op_conf.set_name("inner_product_test");
  InnerProductOpConf* inner_product_conf = op_conf.mutable_innerproduct_conf();
  inner_product_conf->set_in("ip_in");
  inner_product_conf->set_out("ip_out");
  inner_product_conf->set_out_num(40);
  inner_product_conf->set_has_bias_term(has_bias_term);
  auto inner_product_op = OpMgr::Singleton()->ConstructOp(op_conf);

  OperatorProto op_proto;
  inner_product_op->ToProto(&op_proto);

  auto inner_product_kernel =
      new InnerProductKernel<device_type, FloatingPointType>();
  inner_product_kernel->InitFromOpProto(op_proto);

  return inner_product_kernel;
}

template<DeviceType device_type>
void SyncStream(KernelCtx* ctx);

template<>
void SyncStream<DeviceType::kCPU>(KernelCtx* ctx) {
  ctx->device_ctx->cpu_stream()->CloseSendEnd();

  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    while (ctx->device_ctx->cpu_stream()->ReceiveWork(&work) == 0) { work(); }
  });
  cpu_thread.join();
}

template<>
void SyncStream<DeviceType::kGPU>(KernelCtx* ctx) {
  CHECK_EQ(cudaStreamSynchronize(ctx->device_ctx->cuda_stream()), cudaSuccess);
}

template<typename FloatingPointType>
void BlobCmpCpu(Blob* lhs, Blob* rhs) {
  const FloatingPointType* dptr_lhs =
      static_cast<const FloatingPointType*>(lhs->dptr());
  const FloatingPointType* dptr_rhs =
      static_cast<const FloatingPointType*>(rhs->dptr());
  size_t dptr_size = lhs->shape().elem_cnt();

  for (size_t i = 0; i < dptr_size; ++i) {
    ASSERT_FLOAT_EQ(dptr_lhs[i], dptr_rhs[i]);
  }
}

template<typename FloatingPointType>
void BlobCmpGpu(Blob* lhs, Blob* rhs) {
  FloatingPointType* dptr;
  size_t dptr_size = lhs->shape().elem_cnt() * sizeof(FloatingPointType);
  cudaMallocHost(&dptr, dptr_size);
  memset(dptr, 0, dptr_size);

  Blob* copy_lhs = CreateBlob<FloatingPointType>(lhs->shape().dim_vec(), dptr,
                                                 Location::kHost);
  Blob* copy_rhs = CreateBlob<FloatingPointType>(rhs->shape().dim_vec(), dptr,
                                                 Location::kHost);

  cudaMemcpy(copy_lhs->mut_dptr(), lhs->dptr(), dptr_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(copy_rhs->mut_dptr(), rhs->dptr(), dptr_size,
             cudaMemcpyDeviceToHost);

  BlobCmpCpu<FloatingPointType>(copy_lhs, copy_rhs);
}

template<typename FloatingPointType>
void CheckResult(std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
                 std::function<void(Blob*, Blob*)> CmpFunc,
                 bool has_bias_term) {
  CmpFunc(BnInOp2BlobPtr("out"), BnInOp2BlobPtr("expected_out"));
  CmpFunc(BnInOp2BlobPtr("in_diff"), BnInOp2BlobPtr("expected_in_diff"));
  CmpFunc(BnInOp2BlobPtr("weight_diff"),
          BnInOp2BlobPtr("expected_weight_diff"));

  if (has_bias_term) {
    CmpFunc(BnInOp2BlobPtr("bias_diff"), BnInOp2BlobPtr("expected_bias_diff"));
  }
}

template<DeviceType device_type, typename FloatingPointType>
void TestInnerProductKernel(bool has_bias_term) {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2BlobPtr =
      BuildBnInOp2BlobPtr<device_type, FloatingPointType>(has_bias_term);

  auto inner_product_kernel =
      BuildInnerProductKernel<device_type, FloatingPointType>(has_bias_term);

  inner_product_kernel->Forward(ctx, BnInOp2BlobPtr);
  inner_product_kernel->Backward(ctx, BnInOp2BlobPtr);

  SyncStream<device_type>(&ctx);

  if (device_type == DeviceType::kCPU) {
    CheckResult<FloatingPointType>(
        BnInOp2BlobPtr, BlobCmpCpu<FloatingPointType>, has_bias_term);
  } else {
    CheckResult<FloatingPointType>(
        BnInOp2BlobPtr, BlobCmpGpu<FloatingPointType>, has_bias_term);
  }
}

}  // namespace

TEST(InnerProductKernel, inner_product_kernel_cpu_with_bias) {
  TestInnerProductKernel<DeviceType::kCPU, float>(true);
  TestInnerProductKernel<DeviceType::kCPU, double>(true);
}

TEST(InnerProductKernel, inner_product_kernel_cpu_without_bias) {
  TestInnerProductKernel<DeviceType::kCPU, float>(false);
  TestInnerProductKernel<DeviceType::kCPU, double>(false);
}

TEST(InnerProductKernel, inner_product_kernel_gpu_with_bias) {
  TestInnerProductKernel<DeviceType::kGPU, float>(true);
  TestInnerProductKernel<DeviceType::kGPU, double>(true);
}

TEST(InnerProductKernel, inner_product_kernel_gpu_without_bias) {
  TestInnerProductKernel<DeviceType::kGPU, float>(false);
  TestInnerProductKernel<DeviceType::kGPU, double>(false);
}

}  // namespace oneflow
