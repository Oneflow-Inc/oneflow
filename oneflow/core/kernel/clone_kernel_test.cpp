#include <random>
#include "oneflow/core/kernel/clone_kernel.h"
#include "oneflow/core/actor/cpu_device_context.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {

namespace {

enum class Location {
  kHost,
  kDevice
};

struct CloneKernelBlobs {
  CloneKernelBlobs(const int);
  int out_num;
  Blob* in_blob;
  Blob* in_diff_blob;
  std::vector<Blob*> out_blobs;
  std::vector<Blob*> out_diff_blobs;
};

CloneKernelBlobs::CloneKernelBlobs(const int out_num_param):
  out_num(out_num_param),
  in_blob(nullptr), in_diff_blob(nullptr),
  out_blobs(out_num_param, nullptr),
  out_diff_blobs(out_num_param, nullptr) {
}

Blob* CreateBlob(const std::vector<int64_t>& dim_vec,
                 const float* data_vec,
                 Location location) {
  void* dptr = nullptr;
  Shape* shape = new Shape(dim_vec);

  size_t dptr_size = shape->elem_cnt()*sizeof(float);
  if (location == Location::kHost) {
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

Blob* CreateBlobWithSameValue(const std::vector<int64_t>& dim_vec,
                              float value, Location location) {
  Shape* shape = new Shape(dim_vec);
  float* data_vec = new float[shape->elem_cnt()];
  std::fill(data_vec, data_vec+shape->elem_cnt(), value);
  return CreateBlob(dim_vec, data_vec, location);
}

Blob* CreateBlobWithRandomValue(const std::vector<int64_t>& dim_vec,
                                Location location) {
  Shape* shape = new Shape(dim_vec);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 10);
  float* data_vec = new float[shape->elem_cnt()];
  for (int64_t i = 0; i != shape->elem_cnt(); ++i) {
    data_vec[i] = dis(gen);
  }
  return CreateBlob(dim_vec, data_vec, location);
}

OperatorProto BuildCloneOperatorProto(
    const int out_num,
    const std::string& lbn) {
  // Config copy hd operator
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  CloneOpConf* clone_conf = op_conf.mutable_clone_conf();
  clone_conf->set_out_num(out_num);
  clone_conf->set_lbn(lbn);
  auto clone_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  clone_op->ToProto(&op_proto);
  return op_proto;
}

void InitHostCloneKernel(
    CloneKernel<DeviceType::kCPU, float>* clone_kernel,
    const int out_num,
    const std::string& lbn) {
  clone_kernel->InitFromOpProto(BuildCloneOperatorProto(out_num, lbn));
}

void InitDeviceCloneKernel(
    CloneKernel<DeviceType::kGPU, float>* clone_kernel,
    const int out_num,
    const std::string& lbn) {
  clone_kernel->InitFromOpProto(BuildCloneOperatorProto(out_num, lbn));
}

CloneKernelBlobs* CreateCloneKernelBlobsFromData(
    const std::vector<int64_t>& dim_vec,
    const int out_num,
    const HashMap<std::string, float>& init_value_map,
    Location location) {
  CloneKernelBlobs* ck_blobs = new CloneKernelBlobs(out_num);
  ck_blobs->in_blob = CreateBlobWithSameValue(
      dim_vec, init_value_map.at("in_blob"), location);
  ck_blobs->in_diff_blob = CreateBlobWithSameValue(
      dim_vec, init_value_map.at("in_diff_blob"), location);
  // 0 : {k. k. k...}
  // 1 : {2k, 2k, 2k...}
  // ....
  for (int i = 0; i != out_num; ++i) {
    ck_blobs->out_blobs[i] = CreateBlobWithSameValue(
        dim_vec, init_value_map.at("out_blob")*(i+1), location);
    ck_blobs->out_diff_blobs[i] = CreateBlobWithSameValue(
        dim_vec, init_value_map.at("out_diff_blob")*(i+1), location);
  }
  return ck_blobs;
}

CloneKernelBlobs* CreateCloneKernelBlobsWithRandomValue(
    const std::vector<int64_t>& dim_vec,
    const int out_num,
    Location location) {
  CloneKernelBlobs* ck_blobs = new CloneKernelBlobs(out_num);
  ck_blobs->in_blob = CreateBlobWithRandomValue(dim_vec, location);
  ck_blobs->in_diff_blob = CreateBlobWithRandomValue(dim_vec, location);
  for (int i = 0; i != out_num; ++i) {
    ck_blobs->out_blobs[i] = CreateBlobWithSameValue(dim_vec, 0.0f, location);
    ck_blobs->out_diff_blobs[i] = CreateBlobWithRandomValue(dim_vec, location);
  }
  return ck_blobs;
}

void InitBn2BlobPtr(HashMap<std::string, Blob*>& bn2blob_ptr,
                     CloneKernelBlobs* ck_blobs) {
  bn2blob_ptr["in"] = ck_blobs->in_blob;
  bn2blob_ptr["in_diff"] = ck_blobs->in_diff_blob;
  for (size_t i = 0; i != ck_blobs->out_num; ++i) {
    bn2blob_ptr["out_" + std::to_string(i)] = ck_blobs->out_blobs[i];
    bn2blob_ptr["out_" + std::to_string(i) + "_diff"] =
        ck_blobs->out_diff_blobs[i];
  }
}

void CpuStreamExec(int out_num, std::function<Blob*(const std::string&)> fp) {
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(new Channel<std::function<void()>>);
  auto clone_kernel = new CloneKernel<DeviceType::kCPU, float>;
  InitHostCloneKernel(clone_kernel, out_num, "clone_kernel_test");

  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);

  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    // Both Forward and Backward receive out_num times
    for (int i = 0; i < out_num*2; ++i) {
      if (ctx.device_ctx->cpu_stream()->Receive(&work) == 0) {
        work();
      }
    }
  });
  cpu_thread.join();
}

void GpuStreamExec(int out_num, std::function<Blob*(const std::string&)> fp) {
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  cublasHandle_t cublas_handle;
  CHECK_EQ(cublasCreate(&cublas_handle), CUBLAS_STATUS_SUCCESS);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, &cublas_handle, nullptr);
  auto clone_kernel = new CloneKernel<DeviceType::kGPU, float>;
  InitDeviceCloneKernel(clone_kernel, out_num, "clone_kernel_test");

  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);

  CHECK_EQ(cudaStreamSynchronize(ctx.device_ctx->cuda_stream()), cudaSuccess);
}

void TestForward(CloneKernelBlobs* ck_blobs, Location location) {
  const int64_t dptr_size =
      ck_blobs->in_blob->shape().elem_cnt() * sizeof(float);
  void* in_blob_dptr = nullptr;
  std::vector<void*> out_blobs_dptr(ck_blobs->out_num);
  if (location == Location::kHost) {
    in_blob_dptr = ck_blobs->in_blob->mut_dptr();
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_blobs_dptr[i] = ck_blobs->out_blobs[i]->mut_dptr();
    }
  } else {   // Location::kDevice
    in_blob_dptr = malloc(dptr_size);
    CHECK_EQ(cudaMemcpy(in_blob_dptr,
                        ck_blobs->in_blob->dptr(),
                        dptr_size,
                        cudaMemcpyDeviceToHost),
             cudaSuccess);
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_blobs_dptr[i] = malloc(dptr_size);
      CHECK_EQ(cudaMemcpy(out_blobs_dptr[i],
                          ck_blobs->out_blobs[i]->dptr(),
                          dptr_size,
                          cudaMemcpyDeviceToHost),
               cudaSuccess);
    }
  }
  for (int i = 0; i != ck_blobs->out_num; ++i) {
    ASSERT_EQ(memcmp(in_blob_dptr, out_blobs_dptr[i], dptr_size), 0);
  }
}

void TestBackward(CloneKernelBlobs* ck_blobs, Location location) {
  int64_t elem_cnt = ck_blobs->in_blob->shape().elem_cnt();
  int64_t dptr_size = elem_cnt * sizeof(float);
  float* in_diff_dptr = nullptr;
  std::vector<float*> out_diff_dptr(ck_blobs->out_num, nullptr);
  if (location == Location::kHost) {
    in_diff_dptr =
        static_cast<float*>(ck_blobs->in_diff_blob->mut_dptr());
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_diff_dptr[i] =
          static_cast<float*>(ck_blobs->out_diff_blobs[i]->mut_dptr());
    }
  } else {   // Location::kDevice
    void* in_diff_host_cpy = malloc(dptr_size);
    CHECK_EQ(cudaMemcpy(in_diff_host_cpy,
                        ck_blobs->in_diff_blob->dptr(),
                        dptr_size,
                        cudaMemcpyDeviceToHost),
             cudaSuccess);
    std::vector<void*> out_diff_host_cpy(ck_blobs->out_num);
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_diff_host_cpy[i] = malloc(dptr_size);
      CHECK_EQ(cudaMemcpy(out_diff_host_cpy[i],
                          ck_blobs->out_diff_blobs[i]->dptr(),
                          dptr_size,
                          cudaMemcpyDeviceToHost),
               cudaSuccess);
    }
    in_diff_dptr = static_cast<float*>(in_diff_host_cpy);
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_diff_dptr[i] = static_cast<float*>(out_diff_host_cpy[i]);
    }
  }
  for (int64_t i = 0; i != elem_cnt; ++i) {
    float sum = {0.0f};
    for (int out_j = 0; out_j != ck_blobs->out_num; ++out_j) {
      sum += out_diff_dptr[out_j][i];
    }
    ASSERT_FLOAT_EQ(sum, in_diff_dptr[i]);
  }
}

void CloneKernelTest(CloneKernelBlobs* ck_blobs, Location location) {
  HashMap<std::string, Blob*> bn2blob_ptr;
  InitBn2BlobPtr(bn2blob_ptr, ck_blobs);
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };
  if (location == Location::kHost) {
    CpuStreamExec(ck_blobs->out_num, fp);
  } else {
    GpuStreamExec(ck_blobs->out_num, fp);
  }
  TestForward(ck_blobs, location);
  TestBackward(ck_blobs, location);
}

}  // namespace

TEST(CloneKernel, host_random_4x5x6x7) {
  const int out_num = 3;
  const std::vector<int64_t> dim_vec = {4, 5, 6, 7};
  Location location = Location::kHost;
  CloneKernelBlobs* ck_blobs = CreateCloneKernelBlobsWithRandomValue(
      dim_vec, out_num, location);
  CloneKernelTest(ck_blobs, location);
}

TEST(CloneKernel, device_random_4x5x6x7) {
  const int out_num = 3;
  const std::vector<int64_t> dim_vec = {4, 5, 6, 7};
  Location location = Location::kDevice;
  CloneKernelBlobs* ck_blobs = CreateCloneKernelBlobsWithRandomValue(
      dim_vec, out_num, location);
  CloneKernelTest(ck_blobs, location);
}

TEST(CloneKernel, host_fixed_1x3x2) {
  const int out_num = 3;
  const std::vector<int64_t> dim_vec = {1, 3, 2};
  const HashMap<std::string, float> init_value_map = {
    {"in_blob", 1.3}, {"out_blob", 2.4},
    {"in_diff_blob", 3.5}, {"out_diff_blob", 4.6}
  };
  Location location = Location::kHost;
  CloneKernelBlobs* ck_blobs = CreateCloneKernelBlobsFromData(
      dim_vec, out_num, init_value_map, location);
  CloneKernelTest(ck_blobs, location);
}

TEST(CloneKernel, device_fixed_1x3x2) {
  const int out_num = 3;
  const std::vector<int64_t> dim_vec = {1, 3, 2};
  const HashMap<std::string, float> init_value_map = {
    {"in_blob", 1.3}, {"out_blob", 2.4},
    {"in_diff_blob", 3.5}, {"out_diff_blob", 4.6}
  };
  Location location = Location::kDevice;
  CloneKernelBlobs* ck_blobs = CreateCloneKernelBlobsFromData(
      dim_vec, out_num, init_value_map, location);
  CloneKernelTest(ck_blobs, location);
}

}  // namespace oneflow
