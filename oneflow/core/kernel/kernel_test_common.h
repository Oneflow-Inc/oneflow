#ifndef ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_

#include <random>
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

namespace test {

enum class Location { kHost, kDevice };

template<typename FloatingPointType>
Blob* CreateBlobWithVector(const std::vector<int64_t>& dim_vec,
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

template<typename FloatingPointType>
Blob* CreateBlobWithSameValue(const std::vector<int64_t>& dim_vec,
                              FloatingPointType value, Location location) {
  Shape* shape = new Shape(dim_vec);
  FloatingPointType* data_vec = new FloatingPointType[shape->elem_cnt()];
  std::fill(data_vec, data_vec + shape->elem_cnt(), value);
  return CreateBlobWithVector<FloatingPointType>(dim_vec, data_vec, location);
}

template<typename FloatingPointType>
Blob* CreateBlobWithRandomValue(const std::vector<int64_t>& dim_vec,
                                Location location) {
  Shape* shape = new Shape(dim_vec);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<FloatingPointType> dis(0, 10);
  FloatingPointType* data_vec = new FloatingPointType[shape->elem_cnt()];
  for (int64_t i = 0; i != shape->elem_cnt(); ++i) { data_vec[i] = dis(gen); }
  return CreateBlobWithVector<FloatingPointType>(dim_vec, data_vec, location);
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

  Blob* copy_lhs = CreateBlobWithVector<FloatingPointType>(
      lhs->shape().dim_vec(), dptr, Location::kHost);
  Blob* copy_rhs = CreateBlobWithVector<FloatingPointType>(
      rhs->shape().dim_vec(), dptr, Location::kHost);

  cudaMemcpy(copy_lhs->mut_dptr(), lhs->dptr(), dptr_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(copy_rhs->mut_dptr(), rhs->dptr(), dptr_size,
             cudaMemcpyDeviceToHost);

  BlobCmpCpu<FloatingPointType>(copy_lhs, copy_rhs);
}

template<typename FloatingPointType>
void CheckResult(std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
                 const std::string& check, const std::string& expected,
                 std::function<void(Blob*, Blob*)> CmpFunc) {
  CmpFunc(BnInOp2BlobPtr(check), BnInOp2BlobPtr(expected));
}

}  // namespace test
}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
