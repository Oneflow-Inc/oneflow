#include "oneflow/core/kernel/copy_hd_kernel.h"
#include "gtest/gtest.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/cuda_kernel_context.h"
#include "cuda.h"
#include "cuda_runtime.h"

namespace oneflow {

TEST(CopyHdKernel, forward_h2d) {
  std::vector<int64_t> dim_vec = {10, 10, 10, 10};
  Shape* input_shape = new Shape(dim_vec);
  Shape* output_shape = new Shape(dim_vec);

  char* in;
  char* out;
  
  size_t buffer_size = input_shape->elem_cnt()*sizeof(float);
  CHECK_EQ(cudaMallocHost(&in, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMalloc(&out, buffer_size), cudaSuccess);
  
  CHECK_EQ(cudaMemset(in, 0, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMemset(out, 1, buffer_size), cudaSuccess);
  
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  CudaKernelCtx cuda_kernel_ctx(&cuda_stream, nullptr);

  OperatorConf op_conf;
  op_conf.set_name("copy_hd_test");
  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
  
  copy_hd_conf->set_type(CopyHdOpConf::H2D);
  
  OperatorProto op_proto;
  auto copy_hd_op = OpMgr::Singleton().ConstructOp(op_conf);
  copy_hd_op->ToProto(&op_proto);
  
  HashMap<std::string, Blob*> bn2blob_ptr{
       {copy_hd_op->SoleIbn(), new Blob(in, input_shape)},
       {copy_hd_op->SoleObn(), new Blob(out, output_shape)}};
  auto fp = [&bn2blob_ptr](const std::string& bn) {
     return bn2blob_ptr.at(bn);
  };

  CopyHdKernel<DeviceType::kGPU, float> copy_hd_kernel;
  copy_hd_kernel.InitFromOpProto(op_proto);
  copy_hd_kernel.Forward(cuda_kernel_ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);
}

}  // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
