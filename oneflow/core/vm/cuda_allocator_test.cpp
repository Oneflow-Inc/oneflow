#ifdef WITH_CUDA
#include "oneflow/core/vm/cuda_allocator.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {
namespace vm {

TEST(CudaAllocator, cuda_allocator) {
  int gpu_num = -1;
  cudaGetDeviceCount(&gpu_num);
  if (gpu_num <= 0) {
    LOG(INFO) << "CudaAllocator Test: Skip because of non GPU device.";
    return;
  }
  ASSERT_TRUE(cudaSuccess == cudaSetDevice(0));
  size_t free_bytes = -1;
  size_t total_bytes = -1;
  const size_t remain_bytes = 50 * 1048576;
  ASSERT_TRUE(cudaSuccess == cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes <= remain_bytes || free_bytes - remain_bytes < remain_bytes) {
    LOG(INFO) << "CudaAllocator Test: Skip because of allocator mem bytes less than 50MiB in GPU 0";
    return;
  }
  std::unique_ptr<CudaAllocator> allo(new CudaAllocator(0));
  CudaAllocator* a = allo.get();
  std::vector<char*> ptrs;
  for (int i = 0; i < 512; ++i) {
    char* ptr = nullptr;
    a->Allocate(&ptr, 1);
    ASSERT_TRUE(ptr != nullptr);
    ptrs.push_back(ptr);
  }
  std::sort(ptrs.begin(), ptrs.end());
  for (int i = 0; i < 512; ++i) {
    if (i > 0) { ASSERT_TRUE(ptrs.at(i) != ptrs.at(i - 1)); }
    a->Deallocate(ptrs.at(i), 1);
  }

  char* data_ptr_1 = nullptr;
  a->Allocate(&data_ptr_1, 2048 * sizeof(float));
  // float* float_ptr_1 = reinterpret_cast<float*>(data_ptr_1);
  // for (int i = 0; i < 2048; ++i) { *(float_ptr_1 + i) = i * 1.0f; }

  char* data_ptr_2 = nullptr;
  a->Allocate(&data_ptr_2, 4096 * sizeof(double));
  // double* float_ptr_2 = reinterpret_cast<double*>(data_ptr_2);

  ASSERT_TRUE(data_ptr_1 != data_ptr_2);
  if (data_ptr_1 < data_ptr_2) {
    ASSERT_TRUE(data_ptr_1 + 2048 * sizeof(float) <= data_ptr_2);
  } else {
    ASSERT_TRUE(data_ptr_2 + 4096 * sizeof(double) <= data_ptr_1);
  }
  // for (int i = 0; i < 4096; ++i) { *(float_ptr_2 + i) = 4096.0; }

  // ASSERT_TRUE(std::abs((*(float_ptr_1 + 2047)) - 2047.0f) <= 1e-6);
  // ASSERT_TRUE(std::abs((*float_ptr_2) - 4096.0) <= 1e-6);

  a->Deallocate(data_ptr_2, 4096 * sizeof(double));
  a->Deallocate(data_ptr_1, 2048 * sizeof(float));
}

}  // namespace vm
}  // namespace oneflow

#endif  // WITH_CUDA
