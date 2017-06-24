#include <iostream>
#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {

namespace {

enum class Location {
  kHost,
  kDevice
};

Blob* CreateBlob(const std::vector<int64_t>& dim_vec,
    float value, Location dptr_location) {
  void* dptr;
  Shape* shape = new Shape(dim_vec);
  size_t dptr_size = shape->elem_cnt()*sizeof(float);

  // Initialize host memory
  CHECK_EQ(cudaMallocHost(&dptr, dptr_size), cudaSuccess);
  float* fptr = static_cast<float*>(dptr);
  std::fill(fptr, fptr+shape->elem_cnt(), value);

  // copy to device if needed
  if (dptr_location == Location::kDevice) {
    CHECK_EQ(cudaMalloc(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, fptr, dptr_size*sizeof(float),
          cudaMemcpyHostToDevice), cudaSuccess);
  }

  return new Blob(dptr, shape);
}

BoxingKernel<DeviceType::kCPU, float>* BuildBoxingKernel(
    int32_t in_num, int32_t out_num, int kernel_seq,
    BoxingOpConf::InBoxCase in_box_case,
    BoxingOpConf::OutBoxCase out_box_case) {
  // config boxing operator from box cases
  OperatorConf op_conf;
  op_conf.set_name("boxing_test"+std::to_string(kernel_seq));
  BoxingOpConf* boxing_conf = op_conf.mutable_boxing_conf();
  boxing_conf->set_in_num(in_num);
  boxing_conf->set_out_num(out_num);
  if (in_box_case == BoxingOpConf::kConcatBox) {
    boxing_conf->mutable_concat_box()->set_axis(1);
  } else {
    boxing_conf->mutable_add_box();
  }
  if (out_box_case == BoxingOpConf::kDataSplitBox) {
    boxing_conf->mutable_data_split_box();
  } else {
    boxing_conf->mutable_clone_box();
  }

  // Build boxing kernel from configured box
  auto boxing_op = OpMgr::Singleton().ConstructOp(op_conf);
  OperatorProto op_proto;
  boxing_op->ToProto(&op_proto);
  auto boxing_kernel = new BoxingKernel<DeviceType::kCPU, float>;
  boxing_kernel->InitFromOpProto(op_proto);
  return boxing_kernel;
}

std::function<Blob*(const std::string&)>  ConstructBn2BlobPtr(
    const std::vector<std::vector<int64_t> >& in_dim_vecs,
    const std::vector<std::vector<int64_t> >& out_dim_vecs, Location loc) {
  int32_t in_num = in_dim_vecs.size();
  int32_t out_num = out_dim_vecs.size();

  // construct mapping from bns to blobs
  auto bn2blob_ptr = new std::map<std::string, Blob*>;
  for (size_t i=0; i < in_num; ++i) {
    bn2blob_ptr->insert(make_pair("in_" + std::to_string(i),
          CreateBlob(in_dim_vecs[i], (i+1)*1.0, loc)));
    bn2blob_ptr->insert(make_pair("in_" + std::to_string(i) + "_diff",
          CreateBlob(in_dim_vecs[i], 0, loc)));
  }
  for (size_t i=0; i < out_num; ++i) { bn2blob_ptr->insert(make_pair("out_" + std::to_string(i), CreateBlob(out_dim_vecs[i], (i+1)*10.0, loc)));
    bn2blob_ptr->insert(make_pair("out_" + std::to_string(i) + "_diff",
          CreateBlob(out_dim_vecs[i], (i+1)*1.0, loc)));
  }
  bn2blob_ptr->insert(make_pair(std::string("middle"), 
        CreateBlob(in_dim_vecs[0], 0, loc)));

  return [bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr->at(bn);
  };
}

std::function<Blob*(const std::string&)>  ConstructBn2BlobPtr(
    std::function<Blob*(const std::string&)> bn2bptr,
    std::vector<std::vector<int64_t> >& in_dim_vecs,
    std::vector<std::vector<int64_t> >& out_dim_vecs,
    Location loc) {
  auto bn_map = new std::map<std::string, Blob*>;
  // Link the output blobs in bn2bptr, to input blobs in bn_map
  for (size_t i=0; i < out_dim_vecs.size(); ++i) {
    Blob* b = bn2bptr("out_" + std::to_string(i));
    bn_map->insert(make_pair("in_"+std::to_string(i), b));

    b = bn2bptr("out_"+std::to_string(i)+"_diff");
    bn_map->insert(make_pair("in_"+std::to_string(i)+"_diff", b));
  }
  bn_map->insert(make_pair(std::string("middle"), 
        CreateBlob(in_dim_vecs[0], 0, loc)));

  // construct output blobs, the blob numbers should be the same with previous
  // input blobs numbers
  for (size_t i=0; i < in_dim_vecs.size(); ++i) {
    bn_map->insert(make_pair("out_" + std::to_string(i),
          CreateBlob(in_dim_vecs[i], 0, loc)));
    bn_map->insert(make_pair("out_" + std::to_string(i) + "_diff",
          CreateBlob(in_dim_vecs[i], (i+1)*10.0, loc)));
  }

  return [bn_map](const std::string& bn) {
    return bn_map->at(bn);
  };
}

// Mark: will remove in the future
void PrintBlob(Blob* blob) {
  float* fptr = static_cast<float*>(blob->mut_dptr());
  auto dim_vec = blob->shape().dim_vec();
  int a = dim_vec[0], b = dim_vec[1], c=dim_vec[2], d=dim_vec[3];
  printf("Blob size is: %d %d %d %d\n", a, b, c, d);
  float* p = fptr;
  for (size_t i=0; i < a; ++i) {
    for (size_t j=0; j < b; ++j) {
      for (size_t k=0; k < c; ++k) {
        for (size_t z=0; z < d; ++z) {
          printf("%f ", *p++);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}

bool IsBlobEq(Blob* A, Blob* B, Location loc) {
  // check dimension
  std::vector<int64_t> dim_vec_0 = A->shape().dim_vec();
  std::vector<int64_t> dim_vec_1 = B->shape().dim_vec();
  if (dim_vec_0.size() != dim_vec_1.size())
    return false;
  for (size_t i=0; i < dim_vec_0.size(); ++i) {
    if (dim_vec_0.at(i) != dim_vec_1.at(i))
      return false;
  }

  // Move device memory to host if needed 
  size_t data_sz = A->shape().elem_cnt() * sizeof(float);
  const void* dptr_0, *dptr_1;
  if (loc == Location::kDevice) {
    CHECK_EQ(cudaMallocHost(&dptr_0, data_sz), cudaSuccess);
    CHECK_EQ(cudaMallocHost(&dptr_1, data_sz), cudaSuccess);
    CHECK_EQ(cudaMemcpy(&dptr_0, A->dptr(), data_sz,
          cudaMemcpyDeviceToHost), cudaSuccess);
    CHECK_EQ(cudaMemcpy(&dptr_1, B->dptr(), data_sz,
          cudaMemcpyDeviceToHost), cudaSuccess);
  } else {
    dptr_0 = A->dptr();
    dptr_1 = A->dptr();
  }

  // Check blob memory contents
  const char* p = static_cast<const char*>(dptr_0);
  const char* q = static_cast<const char*>(dptr_1);
  for (size_t i=0; i < data_sz; ++i) {
    if (p[i] != q[i]) 
      return false;
  }
  
  return true;
}

}  // namespace


TEST(boxingKernel, boxing_concat_clone_box_cpu) {
  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // Build boxing kernel
  auto boxing_kernel_0 = BuildBoxingKernel(4, 1, 0,
      BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);
  auto boxing_kernel_1 = BuildBoxingKernel(4, 5, 1,
      BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);

  // Build mapping bns->blobs with first kernel
  std::vector<std::vector<int64_t> > in_dim_vecs = { {3, 4, 5, 5},
    {3, 2, 5, 5}, {3, 1, 5, 5}, { 3, 7, 5, 5}};
  std::vector<std::vector<int64_t> > out_dim_vecs_0 = { { 3, 14, 5, 5}};
  std::vector<std::vector<int64_t> > out_dim_vecs_1 = { {3, 14, 5, 5},
    {3, 14, 5, 5}, {3, 14, 5, 5}, {3, 14, 5, 5}, {3, 14, 5, 5} };
  auto fp_0 = ConstructBn2BlobPtr(in_dim_vecs,
      out_dim_vecs_0, Location::kHost); 

  // Build mapping bns->blobs with second kernel
  auto fp_1 = ConstructBn2BlobPtr(in_dim_vecs,
      out_dim_vecs_1, Location::kHost); 
  
  // Run forward && backward test
  boxing_kernel_0->Forward(ctx, fp_0);
  boxing_kernel_1->Forward(ctx, fp_1);
  boxing_kernel_1->Backward(ctx, fp_1);
  boxing_kernel_0->Backward(ctx, fp_0);

  // Check the output blobs 
  for (size_t i=0; i < in_dim_vecs.size(); ++i) {
    const std::string bn = "in_" + std::to_string(i) + "_diff";
  // Mark: The diff should only depend on the first out_0_diff
    ASSERT_TRUE(IsBlobEq(fp_0(bn), fp_1(bn), Location::kHost));
  }
  for (size_t i=0; i < out_dim_vecs_1.size(); ++i) {
    const std::string bn = "out_" + std::to_string(i);
    ASSERT_TRUE(IsBlobEq(fp_1(bn), fp_0("out_0"), Location::kHost));
  }
}

// Mark: code too long and dirty, need refine
// Trick: To test concat and split box, kernel_0 and kernel_1 are connected.
// The data in blobs of inputs of kernel_0 and outputs of kernel_1 
// should be the same.
TEST(boxingKernel, boxing_concat_split_box_cpu) {
  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // Build boxing kernel
  auto boxing_kernel_0 = BuildBoxingKernel(4, 3, 0,
      BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);
  auto boxing_kernel_1 = BuildBoxingKernel(3, 4, 1,
      BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);

  // Build blobs
  std::vector<std::vector<int64_t> > in_dim_vecs = { {3, 4, 5, 5},
    {3, 2, 5, 5}, {3, 1, 5, 5}, { 3, 7, 5, 5}};
  std::vector<std::vector<int64_t> > out_dim_vecs = { {3, 5, 5, 5},
    {3, 6, 5, 5}, {3, 3, 5, 5}};
  auto fp = ConstructBn2BlobPtr(in_dim_vecs,
      out_dim_vecs, Location::kHost); 

  // Build reverse blobs
  auto r_fp = ConstructBn2BlobPtr(fp, in_dim_vecs,
      out_dim_vecs, Location::kHost); 
  
  // Run forward && backward test
  boxing_kernel_0->Forward(ctx, fp);
  boxing_kernel_1->Forward(ctx, r_fp);
  boxing_kernel_1->Backward(ctx, r_fp);
  boxing_kernel_0->Backward(ctx, fp);

  // Check input && output blobs in this graph should be the same
  for (size_t i=0; i < in_dim_vecs.size(); ++i) {
    ASSERT_TRUE(IsBlobEq(fp("in_"+std::to_string(i)),
          r_fp("out_"+std::to_string(i)), Location::kHost));
    ASSERT_TRUE(IsBlobEq(fp("in_"+std::to_string(i)+"_diff"),
          r_fp("out_"+std::to_string(i)+"_diff"), Location::kHost));
  } 
}

TEST(boxingKernel, boxing_add_clone_box_cpu) {
  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // Build boxing kernel
  auto boxing_kernel = BuildBoxingKernel(4, 3, 0, BoxingOpConf::kAddBox,
      BoxingOpConf::kCloneBox);

  // Build mapping bns->blobs
  std::vector<std::vector<int64_t> > in_dim_vecs = { {3, 4, 5, 5},
    {3, 4, 5, 5}, {3, 4, 5, 5}, { 3, 4, 5, 5} };
  std::vector<std::vector<int64_t> > out_dim_vecs = { {3, 4, 5, 5},
    {3, 4, 5, 5}, {3, 4, 5, 5} };
  auto fp = ConstructBn2BlobPtr(in_dim_vecs,
      out_dim_vecs, Location::kHost); 

  // Run forward && backward
  boxing_kernel->Forward(ctx, fp);

  // check if add-results is the same as expected.
  Blob* expected_add_b = CreateBlob(out_dim_vecs[0], 10.0, Location::kHost);
  
  for (size_t i=0; i < out_dim_vecs.size(); ++i) {
    ASSERT_TRUE(IsBlobEq(fp("out_"+std::to_string(i)), expected_add_b,
          Location::kHost));
  }
}

TEST(boxingKernel, boxing_add_split_box_cpu) {
  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // Build boxing kernel
  auto boxing_kernel = BuildBoxingKernel(4, 2, 0, BoxingOpConf::kAddBox,
      BoxingOpConf::kDataSplitBox);

  // Build mapping bns->blobs
  std::vector<std::vector<int64_t> > in_dim_vecs = { {3, 4, 5, 5},
    {3, 4, 5, 5}, {3, 4, 5, 5}, { 3, 4, 5, 5} };
  std::vector<std::vector<int64_t> > out_dim_vecs = { {3, 2, 5, 5},
    {3, 2, 5, 5} };
  auto fp = ConstructBn2BlobPtr(in_dim_vecs,
      out_dim_vecs, Location::kHost); 

  // Run forward
  boxing_kernel->Forward(ctx, fp);

  // check if add-results is the same as expected.
  Blob* expected_add_b = CreateBlob(out_dim_vecs[0], 10.0, Location::kHost);
  
  for (size_t i=0; i < out_dim_vecs.size(); ++i) {
    ASSERT_TRUE(IsBlobEq(fp("out_"+std::to_string(i)), expected_add_b,
          Location::kHost));
  }
}
}  // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
