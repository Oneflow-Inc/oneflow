#include "oneflow/core/kernel/clone_kernel.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/actor/cpu_device_context.h"
#include "oneflow/core/operator/clone_op.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/actor/cuda_device_context.h"
#include <vector>

namespace oneflow {

namespace {

enum class Location {
  kHost,
  kDevice
};

Blob* CreateBlob(const std::vector<int64_t>& dim_vec, float value,
    Location dptr_location) {
  void *dptr, *dev_dptr;
  Shape* shape = new Shape(dim_vec);

  size_t dptr_size = shape->elem_cnt()*sizeof(float);
  dptr = malloc(dptr_size);
  memset(dptr, 0, dptr_size);

  float* dptr_float = static_cast<float*>(dptr);
  for (int i = 0; i != shape->elem_cnt(); ++i) {
    dptr_float[i] = value;
  }

  if (dptr_location == Location::kDevice) {
    CHECK_EQ(cudaMalloc(&dev_dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dev_dptr, dptr, dptr_size, cudaMemcpyHostToDevice), cudaSuccess);
    return new Blob(dev_dptr, shape);
  }

  return new Blob(dptr, shape);
}

OperatorProto BuildCloneOperatorProto(const int out_num, const std::string& lbn) {
  // Config copy hd operator
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  CloneOpConf* clone_conf = op_conf.mutable_clone_conf();
  clone_conf->set_out_num(out_num);
  auto clone_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  clone_op->ToProto(&op_proto);
  return op_proto;
}

void BuildHostCloneKernel(CloneKernel<DeviceType::kCPU, float>* clone_kernel,
                       const int out_num, const std::string& lbn) {
  clone_kernel->InitFromOpProto(BuildCloneOperatorProto(out_num, lbn));
}

void BuildDeviceCloneKernel(CloneKernel<DeviceType::kGPU, float>* clone_kernel,
                       const int out_num, const std::string& lbn) {
  clone_kernel->InitFromOpProto(BuildCloneOperatorProto(out_num, lbn));
}

void HostCloneKernelTest(std::vector<int64_t>& dim_vec, Location location) {
  // Build blob for test
  Blob* in_blob = CreateBlob(dim_vec, 1.0, location);
  Blob* in_diff_blob = CreateBlob(dim_vec, 1.0, location);

  const int out_num = 3;
  std::vector<Blob*> out_blobs, out_diff_blobs;
  for (int i = 0; i != out_num; ++i) {
    out_blobs.push_back(CreateBlob(dim_vec, 0.0, location));
    out_diff_blobs.push_back(CreateBlob(dim_vec, 1.0, location));
  }

  // Create CpuDeviceContext and KernelContext
  auto cpu_stream = new Channel<std::function<void()>>();
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(cpu_stream);

  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    for (int i = 0; i < out_num*2; ++i) {
      if(ctx.device_ctx->cpu_stream()->Receive(&work) == 0) {
        work();
      }
    }
  });

  // build CloneKernel
  auto clone_kernel = new CloneKernel<DeviceType::kCPU, float>;
  BuildHostCloneKernel(clone_kernel, out_num, "clone_kernel_test");

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr{
      {"in", in_blob},
      {"in_diff", in_diff_blob}};
  for (int i = 0; i != out_num; ++i) {
    bn2blob_ptr["out_" + std::to_string(i)] = out_blobs[i];
    bn2blob_ptr["out_" + std::to_string(i) + "_diff"] = out_diff_blobs[i];
  }
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  // clone in_blob -> out_blobs
  // clone in_blob -> out_blobs
  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);
  cpu_thread.join();

  // test Forward
  for (int i = 0; i != out_num; ++i) {
    ASSERT_EQ(memcmp(in_blob->dptr(),
                     out_blobs[i]->dptr(),
                     in_blob->shape().elem_cnt() * sizeof(float)),
              0);
  }

  // test Backward
  float* in_diff_dptr = static_cast<float*>(in_diff_blob->mut_dptr());
  float* out0_diff_dptr = static_cast<float*>(out_diff_blobs[0]->mut_dptr());
  for (int i = 0; i != in_diff_blob->shape().elem_cnt(); ++i) {
    ASSERT_EQ(out0_diff_dptr[i] * out_num, in_diff_dptr[i]);
  }
}

void DeviceCloneKernelTest(std::vector<int64_t>& dim_vec, Location location) {
  // Build blob for test
  Blob* in_blob = CreateBlob(dim_vec, 1.0, location);
  Blob* in_diff_blob = CreateBlob(dim_vec, 1.0, location);

  const int out_num = 3;
  std::vector<Blob*> out_blobs, out_diff_blobs;
  for (int i = 0; i != out_num; ++i) {
    out_blobs.push_back(CreateBlob(dim_vec, 0.0, location));
    out_diff_blobs.push_back(CreateBlob(dim_vec, 1.0, location));
  }

  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  cublasHandle_t cublas_handle;
  CHECK_EQ(cublasCreate(&cublas_handle), CUBLAS_STATUS_SUCCESS);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, &cublas_handle, nullptr);

  // build CloneKernel
  auto clone_kernel = new CloneKernel<DeviceType::kGPU, float>;
  BuildDeviceCloneKernel(clone_kernel, out_num, "clone_kernel_test");

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr{
      {"in", in_blob},
      {"in_diff", in_diff_blob}};
  for (int i = 0; i != out_num; ++i) {
    bn2blob_ptr["out_" + std::to_string(i)] = out_blobs[i];
    bn2blob_ptr["out_" + std::to_string(i) + "_diff"] = out_diff_blobs[i];
  }
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  // clone in_blob -> out_blobs
  clone_kernel->Forward(ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(ctx.device_ctx->cuda_stream()), cudaSuccess);

  // test Forward
  int dptr_size = in_blob->shape().elem_cnt() * sizeof(float);

  void* in_blob_dptr = malloc(dptr_size);
  std::vector<void*> out_blobs_dptr(out_num);

  CHECK_EQ(cudaMemcpy(in_blob_dptr, in_blob->dptr(), dptr_size, cudaMemcpyDeviceToHost),
           cudaSuccess);
  for (int i = 0; i != out_num; ++i) {
    out_blobs_dptr[i] = malloc(dptr_size);
    CHECK_EQ(cudaMemcpy(out_blobs_dptr[i], out_blobs[i]->dptr(), dptr_size, cudaMemcpyDeviceToHost),
             cudaSuccess);
  }
  for (int i = 0; i != out_num; ++i) {
    ASSERT_EQ(memcmp(in_blob_dptr,
                     out_blobs_dptr[i],
                     dptr_size),
              0);
  }

  // clone in_blob -> out_blobs
  clone_kernel->Backward(ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(ctx.device_ctx->cuda_stream()), cudaSuccess);

  // test Backward
  void* in_diff_host_cpy = malloc(dptr_size);
  cudaMemcpy(in_diff_host_cpy, in_diff_blob->dptr(), dptr_size, cudaMemcpyDeviceToHost);
  std::vector<void*> out_diff_host_cpy(out_num);
  for(int i = 0; i != out_num; ++i) {
    out_diff_host_cpy[i] = malloc(dptr_size);
    cudaMemcpy(out_diff_host_cpy[i], out_diff_blobs[i]->dptr(), dptr_size, cudaMemcpyDeviceToHost);
  }
  float* in_diff_dptr = static_cast<float*>(in_diff_host_cpy);
  float* out0_diff_dptr = static_cast<float*>(out_diff_host_cpy[0]);
  for (int i = 0; i != in_diff_blob->shape().elem_cnt(); ++i) {
    ASSERT_EQ(out0_diff_dptr[i] * out_num, in_diff_dptr[i]);
  }
}

}  // namespace

TEST(CloneKernel, host_clone_4x5x6x7) {
  std::vector<int64_t> dim_vec = {4, 5, 6, 7};

  HostCloneKernelTest(dim_vec, Location::kHost);
}

TEST(CloneKernel, device_clone_4x5x6x7) {
  std::vector<int64_t> dim_vec = {4, 5, 6, 7};

  DeviceCloneKernelTest(dim_vec, Location::kDevice);
}

}  // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
