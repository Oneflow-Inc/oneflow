#include "oneflow/core/kernel/copy_hd_kernel.h"
#include <memory>
#include "gtest/gtest.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/copy_hd_op.h"
#include "oneflow/core/actor/cuda_device_context.h"
#include "cuda.h"
#include "cuda_runtime.h"

namespace oneflow {

namespace {

enum Location {
  Host = 0,
  Device = 1
};

Blob* CreateBlob(const std::vector<int64_t>& dim_vec,
                 int filled, enum Location flag) { 
  char* buffer;
  Shape* shape = new Shape(dim_vec);
  
  size_t buffer_size = shape->elem_cnt()*sizeof(float);
  if (flag == Host) {
    CHECK_EQ(cudaMallocHost(&buffer, buffer_size), cudaSuccess);
  } else {
    CHECK_EQ(cudaMalloc(&buffer, buffer_size), cudaSuccess);
  }

  CHECK_EQ(cudaMemset(buffer, filled, buffer_size), cudaSuccess);
  return new Blob(buffer, shape);
}

void BuildCopyHdKernel(CopyHdKernel<DeviceType::kGPU, float>* copy_hd_kernel,
                       enum CopyHdOpConf_Type type) {
  // Config copy hd operator
  OperatorConf op_conf;
  op_conf.set_name("copy_hd_test");
  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
  copy_hd_conf->set_type(type);
  auto copy_hd_op = OpMgr::Singleton().ConstructOp(op_conf);
  
  OperatorProto op_proto;
  copy_hd_op->ToProto(&op_proto);
  copy_hd_kernel->InitFromOpProto(op_proto);
}

}  // namespace

TEST(CopyHdKernel, copy_h2d_3x4x5x6) {
  std::vector<int64_t> dim_vec = {3, 4, 5, 6};
 
  // Build blob for test h2d 
  Blob* blob_host = CreateBlob(dim_vec, 0, Host);
  Blob* blob_device = CreateBlob(dim_vec, 1, Device);
  Blob* check_blob_host = CreateBlob(dim_vec, 2, Host);
 
  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // build CopyH2DKernel
  CopyHdKernel<DeviceType::kGPU, float>* copy_h2d_kernel = 
      new CopyHdKernel<DeviceType::kGPU, float>;
  BuildCopyHdKernel(copy_h2d_kernel, CopyHdOpConf::H2D);
  
  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr{
      {"in", blob_host},
      {"out", blob_device},
      {"in_diff", check_blob_host},
      {"out_diff", blob_device}};
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };
  
  // test forward and backward
  copy_h2d_kernel->Forward(ctx, fp);
  copy_h2d_kernel->Backward(ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);

  // check 
  CHECK_EQ(strcmp(blob_host->dptr(), check_blob_host->dptr()), 0);
}

TEST(CopyHdKernel, copy_d2h_4x5x6x7) {
  std::vector<int64_t> dim_vec = {4, 5, 6, 7};

  // Build blob for test d2h
  Blob* blob_device = CreateBlob(dim_vec, 0, Device);
  Blob* blob_host = CreateBlob(dim_vec, 1, Host);
  Blob* check_blob_device = CreateBlob(dim_vec, 2, Device);
  Blob* check_blob_host = CreateBlob(dim_vec, 3, Host);

  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // build CopyD2HKernel
  CopyHdKernel<DeviceType::kGPU, float>* copy_d2h_kernel =
      new CopyHdKernel<DeviceType::kGPU, float>;
  BuildCopyHdKernel(copy_d2h_kernel, CopyHdOpConf::D2H);

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr{
      {"in", blob_device},
      {"out", blob_host},
      {"in_diff", check_blob_device},
      {"out_diff", blob_host}};
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  }; 

  // test forward and backward
  copy_d2h_kernel->Forward(ctx, fp);
  copy_d2h_kernel->Backward(ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);

  // check
  CHECK_EQ(cudaMemcpy(check_blob_host->mut_dptr(),
                      check_blob_device->dptr(),
                      check_blob_device->shape().elem_cnt() * sizeof(float),
                      cudaMemcpyDeviceToHost),
           cudaSuccess);
  CHECK_EQ(strcmp(blob_host->dptr(), check_blob_host->dptr()), 0);
}

}  // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
