#include "oneflow/core/kernel/copy_hd_kernel.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {

namespace {

enum class Location {
  kHost,
  kDevice
};

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

void BuildCopyHdKernel(CopyHdKernel<DeviceType::kGPU, float>* copy_hd_kernel,
                       CopyHdOpConf::Type hd_type) {
  // Config copy hd operator
  OperatorConf op_conf;
  op_conf.set_name("copy_hd_test");
  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
  copy_hd_conf->set_type(hd_type);
  auto copy_hd_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  copy_hd_op->ToProto(&op_proto);
  copy_hd_kernel->InitFromOpProto(op_proto);
}

}  // namespace

TEST(CopyHdKernel, copy_h2d_3x4x5x6) {
  std::vector<int64_t> dim_vec = {3, 4, 5, 6};

  // Build blob for test h2d
  Blob* blob_host = CreateBlob(dim_vec, 1, Location::kHost);
  Blob* blob_device = CreateBlob(dim_vec, 2, Location::kDevice);
  Blob* expected_blob_host = CreateBlob(dim_vec, 3, Location::kHost);

  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // build CopyH2DKernel
  auto copy_h2d_kernel = new CopyHdKernel<DeviceType::kGPU, float>;
  BuildCopyHdKernel(copy_h2d_kernel, CopyHdOpConf::H2D);

  // Build function pointer of blob name to blob
  HashMap<std::string, Blob*> bn2blob_ptr{
      {"in", blob_host},
      {"out", blob_device},
      {"in_diff", expected_blob_host},
      {"out_diff", blob_device}};
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  // in(blob_host) -> out(blob_device)
  copy_h2d_kernel->Forward(ctx, fp);
  // out_diff(blob_device) -> in_diff(expected_blob_host)
  copy_h2d_kernel->Backward(ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);

  ASSERT_EQ(memcmp(blob_host->dptr(),
                   expected_blob_host->dptr(),
                   blob_host->shape().elem_cnt() * sizeof(float)),
            0);
}

TEST(CopyHdKernel, copy_d2h_4x5x6x7) {
  std::vector<int64_t> dim_vec = {4, 5, 6, 7};

  // Build blob for test d2h
  Blob* blob_device = CreateBlob(dim_vec, 1, Location::kDevice);
  Blob* blob_host = CreateBlob(dim_vec, 2, Location::kHost);
  Blob* expected_blob_device = CreateBlob(dim_vec, 3, Location::kDevice);

  // copy for check
  Blob* blob_device_copy = CreateBlob(dim_vec, 4, Location::kHost);
  Blob* expected_blob_device_copy = CreateBlob(dim_vec, 5, Location::kHost);

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
      {"in_diff", expected_blob_device},
      {"out_diff", blob_host}};
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };

  // in(blob_device) -> out(blob_host)
  copy_d2h_kernel->Forward(ctx, fp);
  // out_diff(blob_host) -> in_diff(expected_blob_device)
  copy_d2h_kernel->Backward(ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);

  // Copy blob_device and expected_blob_device to host for check.
  CHECK_EQ(cudaMemcpy(blob_device_copy->mut_dptr(),
                      blob_device->dptr(),
                      blob_device->shape().elem_cnt() * sizeof(float),
                      cudaMemcpyDeviceToHost),
           cudaSuccess);
  CHECK_EQ(cudaMemcpy(expected_blob_device_copy->mut_dptr(),
                      expected_blob_device->dptr(),
                      expected_blob_device->shape().elem_cnt() * sizeof(float),
                      cudaMemcpyDeviceToHost),
           cudaSuccess);

  ASSERT_EQ(memcmp(blob_device_copy->dptr(),
                   expected_blob_device_copy->dptr(),
                   blob_device_copy->shape().elem_cnt() * sizeof(float)),
            0);
}

}  // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
