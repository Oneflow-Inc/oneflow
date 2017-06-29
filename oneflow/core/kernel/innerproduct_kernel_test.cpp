#include "oneflow/core/kernel/innerproduct_kernel.h"
#include <string>
#include <vector>
#include "oneflow/core/actor/cpu_device_context.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {

namespace {

enum class Location {
  kHost,
  kDevice
};

Blob* CreateBlob(const std::vector<int64_t>& dim_vec, float* matrix,
                 Location mem_location) {
  void* dptr;
  Shape* shape = new Shape(dim_vec);

  size_t dptr_size = shape->elem_cnt()*sizeof(float);
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
Kernel* BuildInnerProductKernel() {
  // Config InnerProduct operator
  OperatorConf op_conf;
  op_conf.set_name("inner_product_test");
  InnerProductOpConf* inner_product_conf = op_conf.mutable_innerproduct_conf();
  inner_product_conf->set_in("ip_in");
  inner_product_conf->set_out("ip_out");
  inner_product_conf->set_out_num(40);
  inner_product_conf->set_has_bias_term(true);
  auto inner_product_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  inner_product_op->ToProto(&op_proto);

  auto inner_product_kernel =
    new InnerProductKernel<device_type, FloatingPointType>();
  inner_product_kernel->InitFromOpProto(op_proto);

  return inner_product_kernel;
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

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  // Build blob for test
  (*bn2blob_ptr)["in"]  = CreateBlob({2, 4}, in_mat, loc);
  (*bn2blob_ptr)["weight"] = CreateBlob({3, 4}, weight_mat, loc);
  (*bn2blob_ptr)["bias"] = CreateBlob({1, 3}, bias_mat, loc);
  (*bn2blob_ptr)["bias_multiplier"] =
    CreateBlob({2, 1}, bias_multiplier_mat, loc);

  (*bn2blob_ptr)["out"] = CreateBlob({2, 3}, out_mat, loc);
  (*bn2blob_ptr)["out_diff"] = (*bn2blob_ptr)["out"];
  (*bn2blob_ptr)["in_diff"] = CreateBlob({2, 4}, in_diff_mat, loc);
  (*bn2blob_ptr)["weight_diff"] = CreateBlob({3, 4}, weight_diff_mat, loc);
  (*bn2blob_ptr)["bias_diff"] = CreateBlob({1, 3}, bias_diff_mat, loc);

  (*bn2blob_ptr)["expected_out"] = CreateBlob({2, 3}, expected_out_mat, loc);
  (*bn2blob_ptr)["expected_in_diff"] =
    CreateBlob({2, 4}, expected_in_diff_mat, loc);
  (*bn2blob_ptr)["expected_weight_diff"] =
    CreateBlob({3, 4}, expected_weight_diff_mat, loc);
  (*bn2blob_ptr)["expected_bias_diff"] =
    CreateBlob({1, 3}, expected_bias_diff_mat, loc);

  return [bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr->at(bn);
  };
}

void BlobCmpCpu(Blob* lhs, Blob* rhs) {
  const float* dptr_lhs = static_cast<const float*>(lhs->dptr());
  const float* dptr_rhs = static_cast<const float*>(rhs->dptr());
  size_t dptr_size = lhs->shape().elem_cnt();

  for (size_t i = 0; i < dptr_size; ++i) {
    ASSERT_FLOAT_EQ(dptr_lhs[i], dptr_rhs[i]);
  }
}

void BlobCmpGpu(Blob* lhs, Blob* rhs) {
  float* dptr;
  size_t dptr_size = lhs->shape().elem_cnt()*sizeof(float);
  cudaMallocHost(&dptr, dptr_size);
  memset(dptr, 0, dptr_size);

  Blob* copy_lhs = CreateBlob(lhs->shape().dim_vec(), dptr, Location::kHost);
  Blob* copy_rhs = CreateBlob(rhs->shape().dim_vec(), dptr, Location::kHost);

  cudaMemcpy(copy_lhs->mut_dptr(), lhs->dptr(), dptr_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(copy_rhs->mut_dptr(), rhs->dptr(), dptr_size,
             cudaMemcpyDeviceToHost);

  BlobCmpCpu(copy_lhs, copy_rhs);
}

void CheckResult(std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
                 std::function<void(Blob*, Blob*)> CmpFunc) {
  CmpFunc(BnInOp2BlobPtr("out"), BnInOp2BlobPtr("expected_out"));
  CmpFunc(BnInOp2BlobPtr("in_diff"), BnInOp2BlobPtr("expected_in_diff"));
  CmpFunc(
      BnInOp2BlobPtr("weight_diff"), BnInOp2BlobPtr("expected_weight_diff"));
  CmpFunc(BnInOp2BlobPtr("bias_diff"), BnInOp2BlobPtr("expected_bias_diff"));
}

}  // namespace

TEST(InnerProductKernel, inner_product_kernel_cpu) {
  // Create cpu_device_context and kernel context
  auto cpu_stream = new Channel<std::function<void()>>;
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(cpu_stream);

  // build InnerProductKernel
  auto inner_product_cpu_kernel =
    BuildInnerProductKernel<DeviceType::kCPU, float>();

  // build function pointer of blob name to blob
  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr(Location::kHost);

  inner_product_cpu_kernel->Forward(ctx, BnInOp2BlobPtr);
  inner_product_cpu_kernel->Backward(ctx, BnInOp2BlobPtr);
  
  ctx.device_ctx->cpu_stream()->CloseSendEnd();

  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    while (ctx.device_ctx->cpu_stream()->Receive(&work) == 0) {
        work();
    }
  });
  cpu_thread.join();

  CheckResult(BnInOp2BlobPtr, BlobCmpCpu);
}

TEST(InnerProductKernel, inner_product_kernel_gpu) {
  // Create gpu_device_context and kernel context
  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  CHECK_EQ(cublasCreate(&cublas_handle), CUBLAS_STATUS_SUCCESS);
  CHECK_EQ(cublasSetStream(cublas_handle, cuda_stream), CUBLAS_STATUS_SUCCESS);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, &cublas_handle, nullptr);

  // Build InnerProductKernel
  auto inner_product_gpu_kernel =
    BuildInnerProductKernel<DeviceType::kGPU, float>();

  // Build function pointer of blob name to blob
  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr(Location::kDevice);

  inner_product_gpu_kernel->Forward(ctx, BnInOp2BlobPtr);
  inner_product_gpu_kernel->Backward(ctx, BnInOp2BlobPtr);

  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);

  CheckResult(BnInOp2BlobPtr, BlobCmpGpu);
}

}  // namespace oneflow
