#include "oneflow/core/kernel/copy_hd_kernel.h"
#include "gtest/gtest.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/cuda_kernel_context.h"
#include "cuda.h"
#include "cuda_runtime.h"

namespace oneflow {

TEST(CopyHdKernel, copy_h2d) {
  // Create buffer pointer and Shape of in_blob and out_blob
  char* in;   // buffer of input blob at host
  char* out;  // buffer of output blob at device
  char* in_check;
  
  char* out_diff;  // buffer of output blob diff at device 
  char* in_diff;   // buffer of input blob diff at host
  char* out_diff_check;
  
  std::vector<int64_t> dim_vec = {10, 10, 10, 10};
  Shape* input_shape    = new Shape(dim_vec);
  Shape* output_shape   = new Shape(dim_vec);
  Shape* in_diff_shape  = new Shape(dim_vec);
  Shape* out_diff_shape = new Shape(dim_vec);
  size_t buffer_size = input_shape->elem_cnt()*sizeof(float);

  // Malloc and Fill memory in host and device
  CHECK_EQ(cudaMallocHost(&in, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMalloc(&out, buffer_size), cudaSuccess); 
  CHECK_EQ(cudaMallocHost(&in_check, buffer_size), cudaSuccess);

  CHECK_EQ(cudaMemset(in, 0, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMemset(out, 1, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMemset(in_check, 1, buffer_size), cudaSuccess);
  
  CHECK_EQ(cudaMalloc(&out_diff, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMallocHost(&in_diff, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMallocHost(&out_diff_check, buffer_size), cudaSuccess);

  CHECK_EQ(cudaMemset(out_diff, 0, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMemset(in_diff, 1, buffer_size), cudaSuccess);
  CHECK_EQ(cudaMemset(out_diff_check, 1, buffer_size), cudaSuccess); 

  // Create CudaKernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  CudaKernelCtx cuda_kernel_ctx(&cuda_stream, nullptr, nullptr);

  // Config copy hd operator and generate op_proto
  OperatorConf op_conf;
  op_conf.set_name("copy_hd_test");
  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
  copy_hd_conf->set_type(CopyHdOpConf::H2D);
  
  OperatorProto op_proto;
  auto copy_hd_op = OpMgr::Singleton().ConstructOp(op_conf);
  copy_hd_op->ToProto(&op_proto);
  
  // Build function pointer of blob name to blob 
  HashMap<std::string, Blob*> bn2blob_ptr{
       {copy_hd_op->SoleIbn(), new Blob(in, input_shape)},
       {copy_hd_op->SoleObn(), new Blob(out, output_shape)},
       {copy_hd_op->SoleOdbn(), new Blob(out_diff, out_diff_shape)},
       {copy_hd_op->SoleIdbn(), new Blob(in_diff, in_diff_shape)}};
  auto fp = [&bn2blob_ptr](const std::string& bn) {
     return bn2blob_ptr.at(bn);
  };

  // build CopyHdKernel
  CopyHdKernel<DeviceType::kGPU, float> copy_hd_kernel;
  copy_hd_kernel.InitFromOpProto(op_proto);
  
  // test forward
  copy_hd_kernel.Forward(cuda_kernel_ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);
  CHECK_EQ(cudaMemcpy(in_check, out, buffer_size,
                      cudaMemcpyDeviceToHost),
           cudaSuccess);
  CHECK_EQ(strcmp(in, in_check), 0);

  // test backward
  copy_hd_kernel.Backward(cuda_kernel_ctx, fp);
  CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);
  CHECK_NE(strcmp(in_diff, out_diff_check), 0);
}

}  // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
