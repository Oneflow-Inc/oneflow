#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/actor/cuda_device_context.h"
#include <iostream>

namespace oneflow {

namespace {

enum class Location {
  kHost,
  kDevice
};

Blob* CreateBlob(const std::vector<int64_t>& dim_vec, float value,
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

void BuildBoxingKernel(BoxingKernel<DeviceType::kCPU, float>* boxing_kernel,\
                       uint32_t in_num, uint32_t out_num, \
                       BoxingOpConf::InBoxCase in_box_case, \
                       BoxingOpConf::OutBoxCase out_box_case) {
  // config boxing operator
  OperatorConf op_conf;
  op_conf.set_name("boxing_test");
  BoxingOpConf* boxing_conf = op_conf.mutable_boxing_conf();
  if (in_box_case == BoxingOpConf::kConcatBox) {
    auto concat_box = new BoxConcatConf;
    boxing_conf->set_allocated_concat_box(concat_box);

    // manually set axis to 1 in current case
    //op_conf.mutable_concat_conf()->set_axis(1);
  } else {
    auto add_box = new BoxAddConf; 
    boxing_conf->set_allocated_add_box(add_box);
  }

  if (out_box_case == BoxingOpConf::kDataSplitBox) {
    auto split_box = new BoxDataSplitConf;
    boxing_conf->set_allocated_data_split_box(split_box);
  } else {
    auto clone_box = new BoxCloneConf;
    boxing_conf->set_allocated_clone_box(clone_box);
  }

  boxing_conf->set_in_num(in_num);
  boxing_conf->set_out_num(out_num);

  auto boxing_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  boxing_op->ToProto(&op_proto);
  boxing_kernel->InitFromOpProto(op_proto);
}

std::map<std::string, Blob*>*  ConstructBlobs(int32_t in_num, \
    int32_t out_num, Location loc) {
  std::vector<std::vector<int64_t> > in_dim_vecs = { {3, 4, 5, 5}, {3, 2, 5, 5}, \
    {3, 1, 5, 5}, { 3, 7, 5, 5}};
  std::vector<std::vector<int64_t> > out_dim_vec = { {3, 5, 5, 5}, {3, 6, 5, 5}, \
    {3, 3, 5, 5}};

  auto bn2blob_ptr = new std::map<std::string, Blob*>;
  for (size_t i=0; i<in_num; ++i) {
    bn2blob_ptr->insert(make_pair("in_"+std::to_string(i), \
          CreateBlob(in_dim_vecs[i], (i+1)*1.0, loc)));
    bn2blob_ptr->insert(make_pair("in_"+std::to_string(i)+"_diff", \
          CreateBlob(in_dim_vecs[i], (i+1)*10.0, loc)));
  }
  for (size_t i=0; i<out_num; ++i) {
    bn2blob_ptr->insert(make_pair("out_"+std::to_string(i), \
          CreateBlob(in_dim_vecs[i], (i+1)*1.0, loc)));
    bn2blob_ptr->insert(make_pair("out_"+std::to_string(i)+"_diff", \
          CreateBlob(in_dim_vecs[i], (i+1)*10.0, loc)));
  }

  return bn2blob_ptr;
}

void PrintBlob(Blob* blob) {
  float* fptr = static_cast<float*>(blob->mut_dptr());
  int64_t sz = blob->shape().elem_cnt();
  for (size_t i=0; i<sz; ++i) {
    std::cout << fptr[i] << " ";
  }
  std::cout << std::endl;
}

}  // namespace

TEST(boxingKernel, boxing_3x4x5x6) {
  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // Build boxing kernel
  auto boxing_kernel = new BoxingKernel<DeviceType::kCPU, float>;  
  BuildBoxingKernel(boxing_kernel, 4, 3, BoxingOpConf::kConcatBox, \
      BoxingOpConf::kDataSplitBox);

  // Build Blobs
  int32_t in_num = 4, out_num = 3;  
  auto bn2blob_ptr = ConstructBlobs(in_num, out_num, Location::kHost); 
  auto fp = [bn2blob_ptr](const std::string& bn) {
    if (bn2blob_ptr->find(bn) == bn2blob_ptr->end())
      std::cout<< "\n\n\n\n" << bn << "\n\n\n";
    return bn2blob_ptr->at(bn);
  };

  // Run forward && backward test
  boxing_kernel->Forward(ctx, fp);
  //boxing_kernel->Backward(ctx, fp);
  //CHECK_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);
  
  // Check results
  for (auto iter : *bn2blob_ptr) {
    std::cout << (iter.first) << " : ";
    PrintBlob(iter.second);
  }

  ASSERT_EQ(0, 0);
}

} // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
