#include "oneflow/core/kernel/clone_kernel.h"
#include "oneflow/core/actor/cpu_device_context.h"
#include "oneflow/core/actor/cuda_device_context.h"
#include <random>

namespace oneflow {

namespace {

enum class Location {
  kHost,
  kDevice
};

void FLOAT_EQ(const float x, const float y) {
  ASSERT_TRUE(abs(x-y) < 0.0001);
}

Blob* CreateBlobFromVector(const std::vector<int64_t>& dim_vec,
                           const std::vector<float>& data_vec,
                           Location location) {
  Shape* shape = new Shape(dim_vec);
  size_t dptr_size = shape->elem_cnt()*sizeof(float);
  void* dptr = malloc(dptr_size);

  float* dptr_float = static_cast<float*>(dptr);
  for (int64_t i = 0; i != shape->elem_cnt(); ++i) {
    dptr_float[i] = data_vec[i];
  }

  if (location == Location::kDevice) {
    void* dev_dptr;
    CHECK_EQ(cudaMalloc(&dev_dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dev_dptr, dptr, dptr_size, cudaMemcpyHostToDevice), cudaSuccess);
    dptr = dev_dptr;
  }
  return new Blob(dptr, shape);
}

Blob* CreateBlobFromValue(const std::vector<int64_t>& dim_vec,
                          float value, Location location) {
  Shape* shape = new Shape(dim_vec);
  const std::vector<float> data_vec(shape->elem_cnt(), value);
  return CreateBlobFromVector(dim_vec, data_vec, location);
}

Blob* CreateRandomBlob(const std::vector<int64_t>& dim_vec, Location location) {
  Shape* shape = new Shape(dim_vec);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 10);
  std::vector<float> data_vec(shape->elem_cnt());
  for (int64_t i = 0; i != shape->elem_cnt(); ++i) {
    data_vec[i] = dis(gen);
  }
  return CreateBlobFromVector(dim_vec, data_vec, location);
}

OperatorProto BuildCloneOperatorProto(const int kOutBlobNum, const std::string& lbn) {
  // Config copy hd operator
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  CloneOpConf* clone_conf = op_conf.mutable_clone_conf();
  clone_conf->set_out_num(kOutBlobNum);
  auto clone_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  clone_op->ToProto(&op_proto);
  return op_proto;
}

void BuildHostCloneKernel(CloneKernel<DeviceType::kCPU, float>* clone_kernel,
                          const int kOutBlobNum, const std::string& lbn) {
  clone_kernel->InitFromOpProto(BuildCloneOperatorProto(kOutBlobNum, lbn));
}

void BuildDeviceCloneKernel(CloneKernel<DeviceType::kGPU, float>* clone_kernel,
                            const int kOutBlobNum, const std::string& lbn) {
  clone_kernel->InitFromOpProto(BuildCloneOperatorProto(kOutBlobNum, lbn));
}

void HostCloneKernelTest(const std::vector<int64_t>& dim_vec,
                         const int kOutBlobNum,
                         const std::vector<float>& data_vec,
                         Location location) {
  // Build blob for test
  Blob* in_blob = CreateBlobFromVector(dim_vec, data_vec, location);
  Blob* in_diff_blob = CreateBlobFromVector(dim_vec, data_vec, location);
  std::vector<Blob*> out_blobs(kOutBlobNum), out_diff_blobs(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_blobs[i] = CreateBlobFromValue(dim_vec, 0.0f, location);
    out_diff_blobs[i] = CreateBlobFromVector(dim_vec, data_vec, location);
  }

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr;
  bn2blob_ptr["in"] = in_blob;
  bn2blob_ptr["in_diff"] = in_diff_blob;
  for (int i = 0; i != kOutBlobNum; ++i) {
    bn2blob_ptr["out_" + std::to_string(i)] = out_blobs[i];
    bn2blob_ptr["out_" + std::to_string(i) + "_diff"] = out_diff_blobs[i];
  }
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  // Create KernelContext and build CloneKernel
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(new Channel<std::function<void()>>);
  auto clone_kernel = new CloneKernel<DeviceType::kCPU, float>;
  BuildHostCloneKernel(clone_kernel, kOutBlobNum, "clone_kernel_test");

  // exec forward and backward
  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);

  // create cpu thread to deal with work received.
  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    for (int i = 0; i < kOutBlobNum*2; ++i) {
      if(ctx.device_ctx->cpu_stream()->Receive(&work) == 0) {
        work();
      }
    }
  });
  cpu_thread.join();

  // test Forward
  for (int i = 0; i != kOutBlobNum; ++i) {
    ASSERT_EQ(memcmp(in_blob->dptr(),
                     out_blobs[i]->dptr(),
                     in_blob->shape().elem_cnt() * sizeof(float)),
              0);
  }
  // test Backward
  float* in_diff_dptr = static_cast<float*>(in_diff_blob->mut_dptr());
  std::vector<float*> out_diff_dptr(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_diff_dptr[i] = static_cast<float*>(out_diff_blobs[i]->mut_dptr());
  }
  for (int64_t i = 0; i != in_diff_blob->shape().elem_cnt(); ++i) {
    float sum = {0.0f};
    for (int out_j = 0; out_j != kOutBlobNum; ++out_j) {
      sum += out_diff_dptr[out_j][i];
    }
    FLOAT_EQ(sum, in_diff_dptr[i]);
  }
}

void HostCloneKernelRandomTest(const std::vector<int64_t>& dim_vec,
                               const int kOutBlobNum,
                               Location location) {
  // Build blob for test
  Blob* in_blob = CreateRandomBlob(dim_vec, location);
  Blob* in_diff_blob = CreateRandomBlob(dim_vec, location);
  std::vector<Blob*> out_blobs(kOutBlobNum), out_diff_blobs(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_blobs[i] = CreateRandomBlob(dim_vec, location);
    out_diff_blobs[i] = CreateRandomBlob(dim_vec, location);
  }

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr;
  bn2blob_ptr["in"] = in_blob;
  bn2blob_ptr["in_diff"] = in_diff_blob;
  for (int i = 0; i != kOutBlobNum; ++i) {
    bn2blob_ptr["out_" + std::to_string(i)] = out_blobs[i];
    bn2blob_ptr["out_" + std::to_string(i) + "_diff"] = out_diff_blobs[i];
  }
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  // Create KernelContext and build CloneKernel
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(new Channel<std::function<void()>>);
  auto clone_kernel = new CloneKernel<DeviceType::kCPU, float>;
  BuildHostCloneKernel(clone_kernel, kOutBlobNum, "clone_kernel_test");

  // exec forward and backward
  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);

  // create cpu thread to deal with work received.
  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    for (int i = 0; i < kOutBlobNum*2; ++i) {
      if(ctx.device_ctx->cpu_stream()->Receive(&work) == 0) {
        work();
      }
    }
  });
  cpu_thread.join();

  // test Forward
  for (int i = 0; i != kOutBlobNum; ++i) {
    ASSERT_EQ(memcmp(in_blob->dptr(),
                     out_blobs[i]->dptr(),
                     in_blob->shape().elem_cnt() * sizeof(float)),
              0);
  }
  // test Backward
  float* in_diff_dptr = static_cast<float*>(in_diff_blob->mut_dptr());
  std::vector<float*> out_diff_dptr(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_diff_dptr[i] = static_cast<float*>(out_diff_blobs[i]->mut_dptr());
  }
  for (int64_t i = 0; i != in_diff_blob->shape().elem_cnt(); ++i) {
    float sum = {0.0f};
    for (int out_j = 0; out_j != kOutBlobNum; ++out_j) {
      sum += out_diff_dptr[out_j][i];
    }
    FLOAT_EQ(sum, in_diff_dptr[i]);
  }
}

void DeviceCloneKernelTest(const std::vector<int64_t>& dim_vec,
                           const int kOutBlobNum,
                           const std::vector<float>& data_vec,
                           Location location) {
  // Build blob for test
  Blob* in_blob = CreateBlobFromVector(dim_vec, data_vec, location);
  Blob* in_diff_blob = CreateBlobFromVector(dim_vec, data_vec, location);

  std::vector<Blob*> out_blobs(kOutBlobNum), out_diff_blobs(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_blobs[i] = CreateBlobFromValue(dim_vec, 0.0f, location);
    out_diff_blobs[i] = CreateBlobFromVector(dim_vec, data_vec, location);
  }

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr;
  bn2blob_ptr["in"] = in_blob;
  bn2blob_ptr["in_diff"] = in_diff_blob;
  for (int i = 0; i != kOutBlobNum; ++i) {
    bn2blob_ptr["out_" + std::to_string(i)] = out_blobs[i];
    bn2blob_ptr["out_" + std::to_string(i) + "_diff"] = out_diff_blobs[i];
  }
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  cublasHandle_t cublas_handle;
  CHECK_EQ(cublasCreate(&cublas_handle), CUBLAS_STATUS_SUCCESS);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, &cublas_handle, nullptr);

  // build CloneKernel
  auto clone_kernel = new CloneKernel<DeviceType::kGPU, float>;
  BuildDeviceCloneKernel(clone_kernel, kOutBlobNum, "clone_kernel_test");

  // exec forward and backward
  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(ctx.device_ctx->cuda_stream()), cudaSuccess);

  // test Forward
  int64_t dptr_size = in_blob->shape().elem_cnt() * sizeof(float);
  void* in_blob_dptr = malloc(dptr_size);
  CHECK_EQ(cudaMemcpy(in_blob_dptr, in_blob->dptr(), dptr_size, cudaMemcpyDeviceToHost),
           cudaSuccess);
  std::vector<void*> out_blobs_dptr(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_blobs_dptr[i] = malloc(dptr_size);
    CHECK_EQ(cudaMemcpy(out_blobs_dptr[i], out_blobs[i]->dptr(), dptr_size, cudaMemcpyDeviceToHost),
             cudaSuccess);
  }
  for (int i = 0; i != kOutBlobNum; ++i) {
    ASSERT_EQ(memcmp(in_blob_dptr,
                     out_blobs_dptr[i],
                     dptr_size),
              0);
  }

  // test Backward
  void* in_diff_host_cpy = malloc(dptr_size);
  cudaMemcpy(in_diff_host_cpy, in_diff_blob->dptr(), dptr_size, cudaMemcpyDeviceToHost);
  std::vector<void*> out_diff_host_cpy(kOutBlobNum);
  for(int i = 0; i != kOutBlobNum; ++i) {
    out_diff_host_cpy[i] = malloc(dptr_size);
    CHECK_EQ(cudaMemcpy(out_diff_host_cpy[i], out_diff_blobs[i]->dptr(), dptr_size, cudaMemcpyDeviceToHost),
             cudaSuccess);
  }
  float* in_diff_dptr = static_cast<float*>(in_diff_host_cpy);
  std::vector<float*> out_diff_dptr(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_diff_dptr[i] = static_cast<float*>(out_diff_host_cpy[i]);
  }
  for (int64_t i = 0; i != in_diff_blob->shape().elem_cnt(); ++i) {
    float sum = {0.0f};
    for (int out_j = 0; out_j != kOutBlobNum; ++out_j) {
      sum += out_diff_dptr[out_j][i];
    }
    FLOAT_EQ(sum, in_diff_dptr[i]);
  }
}

void DeviceCloneKernelRandomTest(const std::vector<int64_t>& dim_vec,
                                 const int kOutBlobNum,
                                 Location location) {
  // Build blob for test
  Blob* in_blob = CreateRandomBlob(dim_vec, location);
  Blob* in_diff_blob = CreateRandomBlob(dim_vec, location);

  std::vector<Blob*> out_blobs(kOutBlobNum), out_diff_blobs(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_blobs[i] = CreateRandomBlob(dim_vec, location);
    out_diff_blobs[i] = CreateRandomBlob(dim_vec, location);
  }

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr;
  bn2blob_ptr["in"] = in_blob;
  bn2blob_ptr["in_diff"] = in_diff_blob;
  for (int i = 0; i != kOutBlobNum; ++i) {
    bn2blob_ptr["out_" + std::to_string(i)] = out_blobs[i];
    bn2blob_ptr["out_" + std::to_string(i) + "_diff"] = out_diff_blobs[i];
  }
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  cublasHandle_t cublas_handle;
  CHECK_EQ(cublasCreate(&cublas_handle), CUBLAS_STATUS_SUCCESS);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, &cublas_handle, nullptr);

  // build CloneKernel
  auto clone_kernel = new CloneKernel<DeviceType::kGPU, float>;
  BuildDeviceCloneKernel(clone_kernel, kOutBlobNum, "clone_kernel_test");

  // exec forward and backward
  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(ctx.device_ctx->cuda_stream()), cudaSuccess);

  // test Forward
  int64_t dptr_size = in_blob->shape().elem_cnt() * sizeof(float);
  void* in_blob_dptr = malloc(dptr_size);
  CHECK_EQ(cudaMemcpy(in_blob_dptr, in_blob->dptr(), dptr_size, cudaMemcpyDeviceToHost),
           cudaSuccess);
  std::vector<void*> out_blobs_dptr(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_blobs_dptr[i] = malloc(dptr_size);
    CHECK_EQ(cudaMemcpy(out_blobs_dptr[i], out_blobs[i]->dptr(), dptr_size, cudaMemcpyDeviceToHost),
             cudaSuccess);
  }
  for (int i = 0; i != kOutBlobNum; ++i) {
    ASSERT_EQ(memcmp(in_blob_dptr,
                     out_blobs_dptr[i],
                     dptr_size),
              0);
  }

  // test Backward
  void* in_diff_host_cpy = malloc(dptr_size);
  cudaMemcpy(in_diff_host_cpy, in_diff_blob->dptr(), dptr_size, cudaMemcpyDeviceToHost);
  std::vector<void*> out_diff_host_cpy(kOutBlobNum);
  for(int i = 0; i != kOutBlobNum; ++i) {
    out_diff_host_cpy[i] = malloc(dptr_size);
    CHECK_EQ(cudaMemcpy(out_diff_host_cpy[i], out_diff_blobs[i]->dptr(), dptr_size, cudaMemcpyDeviceToHost),
             cudaSuccess);
  }
  float* in_diff_dptr = static_cast<float*>(in_diff_host_cpy);
  std::vector<float*> out_diff_dptr(kOutBlobNum);
  for (int i = 0; i != kOutBlobNum; ++i) {
    out_diff_dptr[i] = static_cast<float*>(out_diff_host_cpy[i]);
  }
  for (int64_t i = 0; i != in_diff_blob->shape().elem_cnt(); ++i) {
    float sum = {0.0f};
    for (int out_j = 0; out_j != kOutBlobNum; ++out_j) {
      sum += out_diff_dptr[out_j][i];
    }
    FLOAT_EQ(sum, in_diff_dptr[i]);
  }
}

}  // namespace

TEST(CloneKernel, host_clone_4x5x6x7) {
  const int kOutBlobNum = 3;
  const std::vector<int64_t> dim_vec = {4, 5, 6, 7};
  const std::vector<float> value(1.0, 4*5*6*7);
  HostCloneKernelTest(dim_vec, kOutBlobNum, value, Location::kHost);
}

TEST(CloneKernel, device_clone_4x5x6x7) {
  const int kOutBlobNum = 3;
  const std::vector<int64_t> dim_vec = {4, 5, 6, 7};
  const std::vector<float> value(1.0, 4*5*6*7);
  DeviceCloneKernelTest(dim_vec, kOutBlobNum, value, Location::kDevice);
}

TEST(CloneKernel, host_clone_1x3x2) {
  const int kOutBlobNum = 3;
  const std::vector<int64_t> dim_vec = {1, 3, 2};
  const std::vector<float> value = {1.3, 2.4, 3.5, 4.6, 5.7, 6.8};
  HostCloneKernelTest(dim_vec, kOutBlobNum, value, Location::kHost);
}

TEST(CloneKernel, device_clone_1x3x2) {
  const int kOutBlobNum = 3;
  const std::vector<int64_t> dim_vec = {1, 3, 2};
  const std::vector<float> value = {1.3, 2.4, 3.5, 4.6, 5.7, 6.8};
  DeviceCloneKernelTest(dim_vec, kOutBlobNum, value, Location::kDevice);
}

TEST(CloneKernel, host_clone_4x5x6x7_random) {
  const int kOutBlobNum = 3;
  const std::vector<int64_t> dim_vec = {4, 5, 6, 7};
  const std::vector<float> value(1.0, 4*5*6*7);
  HostCloneKernelRandomTest(dim_vec, kOutBlobNum, Location::kHost);
}

TEST(CloneKernel, device_clone_1x3x2_random) {
  const int kOutBlobNum = 3;
  const std::vector<int64_t> dim_vec = {1, 3, 2};
  const std::vector<float> value = {1.3, 2.4, 3.5, 4.6, 5.7, 6.8};
  DeviceCloneKernelRandomTest(dim_vec, kOutBlobNum, Location::kDevice);
}

}  // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
