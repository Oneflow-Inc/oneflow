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
    CHECK_EQ(cudaMemcpy(dptr, fptr, dptr_size*sizeof(float), \
          cudaMemcpyHostToDevice), cudaSuccess);
  }

  return new Blob(dptr, shape);
}

void BuildBoxingKernel(BoxingKernel<DeviceType::kCPU, float>* boxing_kernel,\
                       uint32_t in_num, uint32_t out_num, int kernel_seq,\
                       BoxingOpConf::InBoxCase in_box_case, \
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
  boxing_kernel->InitFromOpProto(op_proto);
}

std::map<std::string, Blob*>*  ConstructBlobs(
    std::vector<std::vector<int64_t> >& in_dim_vecs,
    std::vector<std::vector<int64_t> >& out_dim_vecs, Location loc) {
  int32_t in_num = in_dim_vecs.size();
  int32_t out_num = out_dim_vecs.size();

  // construct mapping from bns to blobs
  auto bn2blob_ptr = new std::map<std::string, Blob*>;
  for (size_t i=0; i<in_num; ++i) {
    bn2blob_ptr->insert(make_pair("in_" + std::to_string(i), \
          CreateBlob(in_dim_vecs[i], (i+1)*1.0, loc)));
    bn2blob_ptr->insert(make_pair("in_" + std::to_string(i) + "_diff", \
          CreateBlob(in_dim_vecs[i], 0, loc)));
  }
  for (size_t i=0; i<out_num; ++i) {
    bn2blob_ptr->insert(make_pair("out_" + std::to_string(i), \
          CreateBlob(out_dim_vecs[i], (i+1)*10.0, loc)));
    bn2blob_ptr->insert(make_pair("out_" + std::to_string(i) + "_diff", \
          CreateBlob(out_dim_vecs[i], (i+1)*1.0, loc)));
  }

  return bn2blob_ptr;
}

std::map<std::string, Blob*>* ConstructBnMap(\
    std::function<Blob*(const std::string&)> bn2bptr, \
    std::vector<std::vector<int64_t> >& in_dim_vecs, \
    std::vector<std::vector<int64_t> >& out_dim_vecs, \
    Location loc) {
  auto bn_map = new std::map<std::string, Blob*>; 
  // Link the output blobs in bn2bptr, to input blobs in bn_map
  for (size_t i=0; i<out_dim_vecs.size(); ++i) {
    Blob* b = bn2bptr("out_" + std::to_string(i));
    bn_map->insert(make_pair("in_"+std::to_string(i), b));

    b = bn2bptr("out_"+std::to_string(i)+"_diff");
    bn_map->insert(make_pair("in_"+std::to_string(i)+"_diff", b));
  }

  // construct output blobs, the blob numbers should be the same with previous
  // input blobs numbers
  for (size_t i=0; i<in_dim_vecs.size(); ++i) {
    bn_map->insert(make_pair("out_" + std::to_string(i), \
          CreateBlob(in_dim_vecs[i], 0, loc)));
    bn_map->insert(make_pair("out_" + std::to_string(i) + "_diff", \
          CreateBlob(in_dim_vecs[i], (i+1)*10.0, loc)));
  }
  return bn_map;
}

void PrintBlob(Blob* blob) {
  float* fptr = static_cast<float*>(\
      blob->mut_dptr());
  auto dim_vec = blob->shape().dim_vec();
  int a =dim_vec[0], b = dim_vec[1], c=dim_vec[2], d=dim_vec[3];
  printf("Blob size is: %d %d %d %d\n", a, b, c, d);
  float* p = fptr;
  for (size_t i=0; i<a; ++i) {
    for (size_t j=0; j<b; ++j) {
      for (size_t k=0; k<c; ++k) {
        for (size_t z=0; z<d; ++z) {
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
  for (size_t i=0; i<dim_vec_0.size(); ++i) {
    if (dim_vec_0.at(i) != dim_vec_1.at(i))
      return false;
  }
  
  // Move device memory to host if needed 
  size_t data_sz = A->shape().elem_cnt() * sizeof(float);
  const void* dptr_0, *dptr_1;
  if (loc == Location::kDevice) {
    CHECK_EQ(cudaMallocHost(&dptr_0, data_sz), cudaSuccess);
    CHECK_EQ(cudaMallocHost(&dptr_1, data_sz), cudaSuccess);
    CHECK_EQ(cudaMemcpy(&dptr_0, A->dptr(), data_sz, \
          cudaMemcpyDeviceToHost), cudaSuccess);
    CHECK_EQ(cudaMemcpy(&dptr_1, B->dptr(), data_sz, \
          cudaMemcpyDeviceToHost), cudaSuccess);
  } else {
    dptr_0 = A->dptr();
    dptr_1 = A->dptr();
  }

  // Check blob memory contents
  const char* p = static_cast<const char*>(dptr_0);
  const char* q = static_cast<const char*>(dptr_1);
  for (size_t i=0; i<data_sz; ++i) {
    if (p[i] != q[i]) 
      return false;
  }
  
  return true;
}

}  // namespace

// Mark: Too long, code need refine
TEST(boxingKernel, boxing_concat_split_box_cpu) {
  // Create CudaDeviceContext and KernelContext
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, nullptr, nullptr);

  // Build boxing kernel
  auto boxing_kernel_0 = new BoxingKernel<DeviceType::kCPU, float>;  
  BuildBoxingKernel(boxing_kernel_0, 4, 3, 0, BoxingOpConf::kConcatBox, \
      BoxingOpConf::kDataSplitBox);

  auto boxing_kernel_1 = new BoxingKernel<DeviceType::kCPU, float>;  
  BuildBoxingKernel(boxing_kernel_1, 3, 4, 1, BoxingOpConf::kConcatBox, \
      BoxingOpConf::kDataSplitBox);

  // Build blobs
  std::vector<std::vector<int64_t> > in_dim_vecs = { {3, 4, 5, 5}, \
    {3, 2, 5, 5}, {3, 1, 5, 5}, { 3, 7, 5, 5}};
  std::vector<std::vector<int64_t> > out_dim_vecs = { {3, 5, 5, 5}, \
    {3, 6, 5, 5}, {3, 3, 5, 5}};
  auto bn2blob_ptr = ConstructBlobs(in_dim_vecs, \
      out_dim_vecs, Location::kHost); 
  auto fp = [bn2blob_ptr](const std::string& bn) {
    //ASSERT_TRUE(bn2blob_ptr->find(bn) != bn2blob_ptr->end());
    if (bn2blob_ptr->find(bn) == bn2blob_ptr->end()) {
      std::cout << bn << "\n\n\n\n\n";
    }
    return bn2blob_ptr->at(bn);
  };

  // Build reverse blobs 
  auto r_bn2blob_ptr = ConstructBnMap(fp, in_dim_vecs, \
      out_dim_vecs, Location::kHost); 
  auto r_fp = [r_bn2blob_ptr](const std::string& bn) {
    //ASSERT_TRUE(r_bn2blob_ptr->find(bn) != r_bn2blob_ptr->end());
    return r_bn2blob_ptr->at(bn);
  };
  
  // Run forward && backward test
  // Data flow: blobs ----> kernel_0 ----> kernel_1 ----> Final Check
  boxing_kernel_0->Forward(ctx, fp);
  boxing_kernel_1->Forward(ctx, r_fp);
  boxing_kernel_1->Backward(ctx, r_fp);
  boxing_kernel_0->Backward(ctx, fp);

  // Check Results
  for (size_t i=0; i<in_dim_vecs.size(); ++i) {
    ASSERT_TRUE(IsBlobEq(fp("in_"+std::to_string(i)), \
          r_fp("out_"+std::to_string(i)), Location::kHost));
    //std::cout << "Passed one blob!\n";
  }

}

} // namespace oneflow

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
