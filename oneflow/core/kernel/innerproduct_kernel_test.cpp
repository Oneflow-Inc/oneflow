#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/innerproduct_kernel.h"
#include "oneflow/core/actor/cpu_device_context.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {

namespace {

Blob* CreateBlob(const std::vector<int64_t>& dim_vec, int value,
                 Location dptr_location) {
  char* dptr;
  Shape* shape = new Shape(dim_vec);

  size_t dptr_size = shape->elem_cnt()*sizeof(float);
  if (dptr_location == Location::kHost) {
    CHECK_EQ(cudaMallocHost(&dptr, dptr_size), cudaSuccess);
    memset(dptr, value, dptr_size);
  } else {
    CHECK_EQ(cudaMalloc(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemset(dptr, value, dptr_size), cudaSuccess);
  }

  return new Blob(dptr, shape);
}

void BuildInnerProductKernel(
    InnerProductKernel<device_type, floating_point_type>* innerproduct_kernel) {
  // Config InnerProduct operator
  OperatorConf op_conf;
  op_conf.set_name("inner_product_test");
  InnerproductOpConf* inner_product_conf = op_conf.mutable_inner_product_conf();
  inner_product_conf->set_in("ip_in");
  inner_product_conf->set_out("ip_out");
  inner_product_conf->set_out_num(40);
  auto inner_product_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  inner_product_op->ToProto(&op_proto);
  inner_product_op->InitFromOpProto(op_proto);
}

}  // namespace

TEST(InnerProductKernel, inner_product_kernel_cpu) {
  std::vector<int64_t> dim_vec = {1000, 3, 256, 256};

  // Build blob for test
  Blob* in = CreateBlob(dim_vec, 1, Location::kHost);
  Blob* out = CreateBlob(dim_vec, 2, Location::kHost);

  Blob* weight = CreateBlob(dim_vec, 3, Location::kHost);
  Blob* bias = CreateBlob(dim_vec, 4, Location::kHost);
  Blob* bias_multiplier = CreateBlob(dim_vec, 5, Location::kHost);

  Blob* in_diff = CreateBlob(dim_vec, 3, Location::kHost);
  Blob* out_diff = CreateBlob(dim_vec, 4, Location::kHost);

  Blob* weight_diff = CreateBlob(dim_vec, 5, Location::kHost);
  Blob* bias_diff = CreateBlob(dim_vec, 6, Location::kHost);

  // Create cpu_device_context and kernel context
  Channel<std::function<void()>> cpu_stream;
  // TODO(shiyuan)
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(&cpu_stream);

  // build InnerProductKernel
  auto inner_product_cpu_kernel =
    new InnerProductKernel<DeviceType::kCPU, float>;
  BuildInnerProductKernel(inner_product_cpu_kernel);

  // build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr{
      {"in", in},
      {"out", out},
      {"weight", weight},
      {"bias", bias},
      {"bias_multiplier", bias_multiplier},
      {"in_diff", in_diff},
      {"out_diff", out_diff},
      {"weight_diff", weight_diff},
      {"bias_diff", bias_diff}};
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  //
  inner_product_cpu_kernel->Forward(ctx, fp);
  inner_product_cpu_kernel->Backward(ctx, fp);

  ASSERT_EQ(in_diff, 0);  // TODO(shiyuan)
  ASSERT_EQ(weight_diff, 0);  // TODO(shiyuan)
  ASSERT_EQ(bias_diff, 0);  // TODO(shiyuan)
}

TEST(InnerProductKernel, inner_product_kernel_gpu) {
  std::vector<int64_t> dim_vec = {3, 4, 5, 6};

  // Build blob for test
  Blob* in = CreateBlob(dim_vec, 1, Location::kDevice);
  Blob* out = CreateBlob(dim_vec, 2, Location::kDevice);

  Blob* weight = CreateBlob(dim_vec, 3, Location::kDevice);
  Blob* bias = CreateBlob(dim_vec, 4, Location::kDevice);
  Blob* bias_multiplier = CreateBlob(dim_vec, 5, Location::kDevice);

  Blob* in_diff = CreateBlob(dim_vec, 3, Location::kDevice);
  Blob* out_diff = CreateBlob(dim_vec, 4, Location::kDevice);

  Blob* weight_diff = CreateBlob(dim_vec, 5, Location::kDevice);
  Blob* bias_diff = CreateBlob(dim_vec, 6, Location::kDevice);

  // Create gpu_device_context and kernel context
  cublasHandle_t cublas_handle;
  CHECK_EQ(cublasCreate(&cublas_handle), CUBLAS_STATUS_SUCCESS);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(nullptr, &cublas_handle, nullptr);

  // Build InnerProductKernel
  auto inner_product_gpu_kernel =
    new InnerProductKernel<DeviceType::kGPU, float>;
  BuildInnerProductKernel(inner_product_gpu_kernel);

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr{
      {"in", in},
      {"out", out},
      {"weight", weight},
      {"bias", bias},
      {"bias_multiplier", bias_multiplier},
      {"in_diff", in_diff},
      {"out_diff", out_diff},
      {"weight_diff", weight_diff},
      {"bias_diff", bias_diff}};
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  //
  inner_product_gpu_kernel->Forward(ctx, fp);
  inner_product_gpu_kernel->Backward(ctx, fp);

  ASSERT_EQ(in_diff, 0);  // TODO(shiyuan)
  ASSERT_EQ(weight_diff, 0);  // TODO(shiyuan)
  ASSERT_EQ(bias_diff, 0);  // TODO(shiyuan)
}

}  // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
