#include "oneflow/core/kernel/model_diff_accumulate_kernel.h"
#include <random>
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

namespace {

enum class Location { kHost, kDevice };

template<typename FloatingPointType>
Blob* CreateBlob(const std::vector<int64_t>& dim_vec,
                 FloatingPointType* data_vec, Location mem_location) {
  void* dptr;
  Shape* shape = new Shape(dim_vec);

  size_t dptr_size = shape->elem_cnt() * sizeof(FloatingPointType);
  if (mem_location == Location::kHost) {
    CHECK_EQ(cudaMallocHost(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, data_vec, dptr_size, cudaMemcpyHostToHost),
             cudaSuccess);
  } else {
    CHECK_EQ(cudaMalloc(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, data_vec, dptr_size, cudaMemcpyHostToDevice),
             cudaSuccess);
  }

  return new Blob(dptr, shape);
}

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  Location loc;
  if (device_type == DeviceType::kCPU) {
    loc = Location::kHost;
  } else {
    loc = Location::kDevice;
  }

  std::vector<int64_t> dim_vec = {2, 4};
  FloatingPointType diff_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  FloatingPointType diff_acc_data[] = {5, 3, 2, 1, 7, 0, 1, 1};

  FloatingPointType expected_data[] = {6, 5, 5, 5, 12, 6, 8, 9};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  (*bn2blob_ptr)["model_diff"] =
      CreateBlob<FloatingPointType>(dim_vec, diff_data, loc);
  (*bn2blob_ptr)["model_diff_acc"] =
      CreateBlob<FloatingPointType>(dim_vec, diff_acc_data, loc);
  (*bn2blob_ptr)["expected_acc"] =
      CreateBlob<FloatingPointType>(dim_vec, expected_data, loc);
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
Kernel* BuildMdDiffAccKernel() {
  OperatorConf op_conf;
  op_conf.set_name("model_diff_acc");
  op_conf.mutable_model_diff_acc_conf();
  auto model_diff_acc_op = OpMgr::Singleton()->ConstructOp(op_conf);

  OperatorProto op_proto;
  model_diff_acc_op->ToProto(&op_proto);

  auto model_diff_acc_kernel =
      new MdDiffAccKernel<device_type, FloatingPointType>();
  model_diff_acc_kernel->InitFromOpProto(op_proto);

  return model_diff_acc_kernel;
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
                 std::function<void(Blob*, Blob*)> CmpFunc) {
  CmpFunc(BnInOp2BlobPtr("model_diff_acc"), BnInOp2BlobPtr("expected_acc"));
}

template<DeviceType device_type, typename FloatingPointType>
void TestMdDiffAccKernel() {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();

  auto model_diff_acc_kernel =
      BuildMdDiffAccKernel<device_type, FloatingPointType>();

  model_diff_acc_kernel->Forward(ctx, BnInOp2BlobPtr);
  SyncStream<device_type>(&ctx);

  if (device_type == DeviceType::kCPU) {
    CheckResult<FloatingPointType>(BnInOp2BlobPtr,
                                   BlobCmpCpu<FloatingPointType>);
  } else {
    CheckResult<FloatingPointType>(BnInOp2BlobPtr,
                                   BlobCmpGpu<FloatingPointType>);
  }
}
}  // namespace

TEST(MdDiffAccKernel, model_diff_acc_kernel_cpu) {
  TestMdDiffAccKernel<DeviceType::kCPU, float>();
  TestMdDiffAccKernel<DeviceType::kCPU, double>();
}

TEST(MdDiffAccKernel, model_diff_acc_kernel_gpu) {
  TestMdDiffAccKernel<DeviceType::kGPU, float>();
  TestMdDiffAccKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
