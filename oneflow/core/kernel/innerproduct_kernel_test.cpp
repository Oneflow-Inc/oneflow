#include "oneflow/core/kernel/innerproduct_kernel.h"
#include <memory>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/innerproduct_op.h"
#include "oneflow/core/actor/cpu_device_context.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {

namespace {

enum class Location {
  kHost,
  kDevice
};

Blob* CreateBlob(const std::vector<int64_t>& dim_vec, float* mat,
                 Location mem_location) {
  void* dptr;
  Shape* shape = new Shape(dim_vec);

  size_t dptr_size = shape->elem_cnt()*sizeof(float);
  if (mem_location == Location::kHost) {
    CHECK_EQ(cudaMallocHost(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, mat, dptr_size, cudaMemcpyHostToHost),
             cudaSuccess);
  } else {
    CHECK_EQ(cudaMalloc(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, mat, dptr_size, cudaMemcpyHostToDevice),
             cudaSuccess);
  }

  return new Blob(dptr, shape);
}

template<DeviceType device_type, typename floating_point_type>
void BuildInnerProductKernel(
    InnerProductKernel<device_type,
                       floating_point_type>* inner_product_kernel) {
  // Config InnerProduct operator
  OperatorConf op_conf;
  op_conf.set_name("inner_product_test");
  InnerProductOpConf* inner_product_conf = op_conf.mutable_innerproduct_conf();
  inner_product_conf->set_in("ip_in");
  inner_product_conf->set_out("ip_out");
  inner_product_conf->set_out_num(40);
  auto inner_product_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  inner_product_op->ToProto(&op_proto);
  inner_product_kernel->InitFromOpProto(op_proto);
}

std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(Location loc) {
  // Create matrix
  float in_mat[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float weight_mat[] = {5, 4, 5, 3, 2, 1, 7, 0, 1, 1, 9, 8};
  float bias_mat[] = {2, 3, 5};
  float bias_multiplier_mat[] = {1, 1};
  float out_mat[6] = {0};
  float in_diff_mat[8] = {0};
  float weight_diff_mat[12] = {0};
  float bias_diff_mat[3] = {0};
  float expected_out_mat[] = {42, 28, 67, 110, 68, 143};
  float expected_in_diff_mat[] = {333, 263, 1009, 662, 829, 651, 2313, 1474};
  float expected_weight_diff_mat[] = {592, 744,  896, 1048,
                                      368, 464,  560,  656,
                                      782, 992, 1202, 1412};
  float expected_bias_diff_mat[] = {152, 96, 210};

  // Build blob for test
  Blob* in = CreateBlob({2, 4}, in_mat, loc);
  Blob* weight = CreateBlob({3, 4}, weight_mat, loc);
  Blob* bias = CreateBlob({1, 3}, bias_mat, loc);
  Blob* bias_multiplier = CreateBlob({2, 1}, bias_multiplier_mat, loc);
  Blob* out = CreateBlob({2, 3}, out_mat, loc);
  Blob* out_diff = out;
  Blob* in_diff = CreateBlob({2, 4}, in_diff_mat, loc);
  Blob* weight_diff = CreateBlob({3, 4}, weight_diff_mat, loc);
  Blob* bias_diff = CreateBlob({1, 3}, bias_diff_mat, loc);
  Blob* expected_out = CreateBlob({2, 3}, expected_out_mat, loc);
  Blob* expected_in_diff = CreateBlob({2, 4}, expected_in_diff_mat, loc);
  Blob* expected_weight_diff = CreateBlob(
      {3, 4}, expected_weight_diff_mat, loc);
  Blob* expected_bias_diff = CreateBlob({1, 3}, expected_bias_diff_mat, loc);

  auto bn2blob_ptr = new HashMap<std::string, Blob*>{
      {"in", in},
      {"out", out},
      {"weight", weight},
      {"bias", bias},
      {"bias_multiplier", bias_multiplier},
      {"in_diff", in_diff},
      {"out_diff", out_diff},
      {"weight_diff", weight_diff},
      {"bias_diff", bias_diff},
      {"expected_out", expected_out},
      {"expected_in_diff", expected_in_diff},
      {"expected_weight_diff", expected_weight_diff},
      {"expected_bias_diff", expected_bias_diff}};
  return [bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr->at(bn);
  };
}

int BlobCmpCpu(Blob* A, Blob* B) {
  const float* dptr_A = static_cast<const float*>(A->dptr());
  const float* dptr_B = static_cast<const float*>(B->dptr());
  size_t dptr_size = A->shape().elem_cnt();
  float epsilon = 1e-10;
  int ret = 0;

  for (size_t i = 0; i < dptr_size; ++i) {
    if (dptr_A[i] - dptr_B[i] > epsilon) {
      ret = -1;
      return ret;
    } else if (dptr_B[i] - dptr_A[i] > epsilon) {
      ret = 1;
      return ret;
    }
  }
  return ret;
}

int BlobCmpGpu(Blob* A, Blob* B) {
  float* dptr;
  size_t dptr_size = A->shape().elem_cnt()*sizeof(float);
  cudaMallocHost(&dptr, dptr_size);
  memset(dptr, 0, dptr_size);

  Blob* copy_A = CreateBlob(A->shape().dim_vec(), dptr, Location::kHost);
  Blob* copy_B = CreateBlob(B->shape().dim_vec(), dptr, Location::kHost);

  cudaMemcpy(copy_A->mut_dptr(), A->dptr(), dptr_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(copy_B->mut_dptr(), B->dptr(), dptr_size, cudaMemcpyDeviceToHost);

  return BlobCmpCpu(copy_A, copy_B);
}

}  // namespace

TEST(InnerProductKernel, inner_product_kernel_cpu) {
  // Create cpu_device_context and kernel context
  auto cpu_stream = new Channel<std::function<void()>>;
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(cpu_stream);

  // build InnerProductKernel
  auto inner_product_cpu_kernel = new InnerProductKernel<DeviceType::kCPU, float>;
  BuildInnerProductKernel(inner_product_cpu_kernel);

  // build function pointer of blob name to blob
  auto fp = BuildBnInOp2BlobPtr(Location::kHost);

  inner_product_cpu_kernel->Forward(ctx, fp);
  inner_product_cpu_kernel->Backward(ctx, fp);

  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    for (int i = 0; i < 5; ++i) {
      if(ctx.device_ctx->cpu_stream()->Receive(&work) == 0) {
        work();
      }
    }
  });
  cpu_thread.join();
  
  ASSERT_EQ(BlobCmpCpu(fp("out"), fp("expected_out")), 0);
  ASSERT_EQ(BlobCmpCpu(fp("in_diff"), fp("expected_in_diff")), 0);
  ASSERT_EQ(BlobCmpCpu(fp("weight_diff"), fp("expected_weight_diff")), 0);
  ASSERT_EQ(BlobCmpCpu(fp("bias_diff"), fp("expected_bias_diff")), 0);
}

TEST(InnerProductKernel, inner_product_kernel_gpu) {
  // Create gpu_device_context and kernel context
  cublasHandle_t cublas_handle;
  CHECK_EQ(cublasCreate(&cublas_handle), CUBLAS_STATUS_SUCCESS);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(nullptr, &cublas_handle, nullptr);

  // Build InnerProductKernel
  auto inner_product_gpu_kernel = new InnerProductKernel<DeviceType::kGPU, float>;
  BuildInnerProductKernel(inner_product_gpu_kernel);

  // Build function pointer of blob name to blob
  auto fp = BuildBnInOp2BlobPtr(Location::kDevice);

  inner_product_gpu_kernel->Forward(ctx, fp);
  inner_product_gpu_kernel->Backward(ctx, fp);

  ASSERT_EQ(BlobCmpGpu(fp("out"), fp("expected_out")), 0);
  ASSERT_EQ(BlobCmpGpu(fp("in_diff"), fp("expected_in_diff")), 0);
  ASSERT_EQ(BlobCmpGpu(fp("weight_diff"), fp("expected_weight_diff")), 0);
  ASSERT_EQ(BlobCmpGpu(fp("bias_diff"), fp("expected_bias_diff")), 0);
}

}  // namespace oneflow
