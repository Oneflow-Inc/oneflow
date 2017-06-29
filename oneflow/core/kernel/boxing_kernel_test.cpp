#include <iostream>
#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/operator/operator.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/actor/cpu_device_context.h"

namespace oneflow {

namespace {

void FakeRun(Channel<std::function<void()> >* cpu_stream) {
  std::function<void()> work;
  cpu_stream->CloseSendEnd();
  while (cpu_stream->Receive(&work) == 0) {
    work();
  }
}

Blob* CreateBlob(const std::vector<int64_t>& dim_vec, float value) {
  Shape* shape = new Shape(dim_vec);
  size_t dptr_size = shape->elem_cnt() * sizeof(float);

  float* dptr = static_cast<float*>(malloc(dptr_size));
  std::fill(dptr, dptr+shape->elem_cnt(), value);
  
  return new Blob(dptr, shape);
}

void BlobCmp(Blob* A, Blob* B) {
  const float* dptr_A = static_cast<const float*>(A->dptr());
  const float* dptr_B = static_cast<const float*>(B->dptr());
  size_t dptr_size = A->shape().elem_cnt();

  for (size_t i = 0; i < dptr_size; ++i) {
    ASSERT_FLOAT_EQ(dptr_A[i], dptr_B[i]);
  }
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

std::function<Blob*(const std::string&)> ConstructBnInOp2BlobPtr(
    const std::vector<std::vector<int64_t> >& in_dim_vecs,
    const std::vector<std::vector<int64_t> >& out_dim_vecs,
    const std::vector<int64_t> middle_dim={0, 0, 0, 0}) {
  int32_t in_num = in_dim_vecs.size();
  int32_t out_num = out_dim_vecs.size();

  // construct mapping from bns to blobs
  auto bn2blob_ptr = new std::map<std::string, Blob*>;
  for (size_t i=0; i < in_num; ++i) {
    bn2blob_ptr->insert(make_pair("in_" + std::to_string(i),
          CreateBlob(in_dim_vecs[i], (i+1)*1.0)));
    bn2blob_ptr->insert(make_pair("in_" + std::to_string(i) + "_diff",
          CreateBlob(in_dim_vecs[i], 0)));
  }
  for (size_t i=0; i < out_num; ++i) { bn2blob_ptr->insert(
      make_pair("out_" + std::to_string(i), 
        CreateBlob(out_dim_vecs[i], (i+1)*10.0)));
    bn2blob_ptr->insert(make_pair("out_" + std::to_string(i) + "_diff",
          CreateBlob(out_dim_vecs[i], (i+1)*1.0)));
  }
  bn2blob_ptr->insert(make_pair(std::string("middle"), 
        CreateBlob(middle_dim, 0)));

  return [bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr->at(bn);
  };
}

// Use the output blobs in bn2bptr as the input blobs 
std::function<Blob*(const std::string&)> ConstructBnInOp2BlobPtr(
    std::function<Blob*(const std::string&)> bn2bptr,
    const std::vector<std::vector<int64_t> >& in_dim_vecs,
    const std::vector<std::vector<int64_t> >& out_dim_vecs, 
    const std::vector<int64_t> middle_dim = {0, 0, 0, 0}) {
  auto bn2blob_ptr = new std::map<std::string, Blob*>;
  // Link the output blobs in bn2bptr, to input blobs in bn2blob_ptr
  for (size_t i=0; i < out_dim_vecs.size(); ++i) {
    Blob* b = bn2bptr("out_" + std::to_string(i));
    bn2blob_ptr->insert(make_pair("in_"+std::to_string(i), b));

    b = bn2bptr("out_"+std::to_string(i)+"_diff");
    bn2blob_ptr->insert(make_pair("in_"+std::to_string(i)+"_diff", b));
  }
  bn2blob_ptr->insert(make_pair(std::string("middle"), 
        CreateBlob(middle_dim, 0)));

  for (size_t i=0; i < in_dim_vecs.size(); ++i) {
    bn2blob_ptr->insert(make_pair("out_" + std::to_string(i),
          CreateBlob(in_dim_vecs[i], 0)));
    bn2blob_ptr->insert(make_pair("out_" + std::to_string(i) + "_diff",
          CreateBlob(in_dim_vecs[i], (i+1)*10.0))); }

  return [bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr->at(bn);
  };
}

}  // namespace

TEST(boxingKernel, boxing_concat_clone_box) {
  // Create cpu_device and kernel contexts
  auto cpu_stream = new Channel<std::function<void()> >;
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(cpu_stream);

  // Build boxing kernel
  auto boxing_kernel_0 = BuildBoxingKernel(4, 1, 0,
      BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);
  auto boxing_kernel_1 = BuildBoxingKernel(4, 5, 1,
      BoxingOpConf::kConcatBox, BoxingOpConf::kCloneBox);

  // Build mapping bns->blobs with first kernel
  std::vector<std::vector<int64_t> > in_dim_vecs = { {3, 4, 5, 5},
    {3, 2, 5, 5}, {3, 1, 5, 5}, { 3, 7, 5, 5}};
  std::vector<std::vector<int64_t> > out_dim_vecs_0 = { { 3, 14, 5, 5}};
  std::vector<std::vector<int64_t> > out_dim_vecs_1 = { {3, 14, 5, 5},
    {3, 14, 5, 5}, {3, 14, 5, 5}, {3, 14, 5, 5}, {3, 14, 5, 5} };
  
  auto BnInOp2BlobPtr_0 = ConstructBnInOp2BlobPtr(in_dim_vecs,
      out_dim_vecs_0, out_dim_vecs_0[0]); 

  // Build mapping bns->blobs with second kernel
  auto BnInOp2BlobPtr_1 = ConstructBnInOp2BlobPtr(in_dim_vecs,
      out_dim_vecs_1, out_dim_vecs_0[0]); 
  
  // Run forward && backward test
  boxing_kernel_0->Forward(ctx, BnInOp2BlobPtr_0);
  boxing_kernel_1->Forward(ctx, BnInOp2BlobPtr_1);
  boxing_kernel_1->Backward(ctx, BnInOp2BlobPtr_1);
  boxing_kernel_0->Backward(ctx, BnInOp2BlobPtr_0);

  FakeRun(cpu_stream);

  // Check the output blobs 
  Blob* expected_in_diff = CreateBlob({3, 14, 5, 5}, 15.0);
  BlobCmp(BnInOp2BlobPtr_1("middle"), expected_in_diff);

  for (size_t i=0; i < out_dim_vecs_1.size(); ++i) {
    const std::string bn = "out_" + std::to_string(i);
    BlobCmp(BnInOp2BlobPtr_1(bn), BnInOp2BlobPtr_0("out_0"));
  }
}

// Trick: To test concat and split box, kernel_0 and kernel_1 are connected.
// The data in blobs of inputs of kernel_0 and outputs of kernel_1 
// should be the same.
TEST(boxingKernel, boxing_concat_split_box) {
  // Create cpu_device and kernel contexts
  auto cpu_stream = new Channel<std::function<void()>>;
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(cpu_stream);

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
  auto BnInOp2BlobPtr = ConstructBnInOp2BlobPtr(in_dim_vecs,
      out_dim_vecs); 

  // Build reverse blobs
  auto r_BnInOp2BlobPtr = ConstructBnInOp2BlobPtr(BnInOp2BlobPtr, in_dim_vecs,
      out_dim_vecs); 
  
  // Run forward && backward test
  boxing_kernel_0->Forward(ctx, BnInOp2BlobPtr);
  boxing_kernel_1->Forward(ctx, r_BnInOp2BlobPtr);
  boxing_kernel_1->Backward(ctx, r_BnInOp2BlobPtr);
  boxing_kernel_0->Backward(ctx, BnInOp2BlobPtr);

  FakeRun(cpu_stream);

  // Check input && output blobs in this graph should be the same
  for (size_t i=0; i < in_dim_vecs.size(); ++i) {
    BlobCmp(BnInOp2BlobPtr("in_" + std::to_string(i)), 
        r_BnInOp2BlobPtr("out_" + std::to_string(i)));
    BlobCmp(BnInOp2BlobPtr("in_" + std::to_string(i) + "_diff"),
        r_BnInOp2BlobPtr("out_" + std::to_string(i) + "_diff"));
  } 
}

TEST(boxingKernel, boxing_add_clone_box) {
  // Create cpu_device and kernel contexts
  auto cpu_stream = new Channel<std::function<void()>>;
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(cpu_stream);

  // Build boxing kernel
  auto boxing_kernel = BuildBoxingKernel(4, 3, 0, BoxingOpConf::kAddBox,
      BoxingOpConf::kCloneBox);

  // Build mapping bns->blobs
  std::vector<std::vector<int64_t> > in_dim_vecs = { {3, 4, 5, 5},
    {3, 4, 5, 5}, {3, 4, 5, 5}, { 3, 4, 5, 5} };
  std::vector<std::vector<int64_t> > out_dim_vecs = { {3, 4, 5, 5},
    {3, 4, 5, 5}, {3, 4, 5, 5} };
  auto BnInOp2BlobPtr = ConstructBnInOp2BlobPtr(in_dim_vecs,
      out_dim_vecs, {3, 4, 5, 5}); 

  // Run forward && backward
  boxing_kernel->Forward(ctx, BnInOp2BlobPtr);

  FakeRun(cpu_stream);

  // check if add-results is the same as expected.
  Blob* expected_add_b = CreateBlob(out_dim_vecs[0], 10.0);
  
  for (size_t i=0; i < out_dim_vecs.size(); ++i) {
    BlobCmp(BnInOp2BlobPtr("out_"+std::to_string(i)), expected_add_b);
  }
}

}  // namespace oneflow
