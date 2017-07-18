#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<typename FloatingPointType>
BoxingKernel<DeviceType::kCPU, FloatingPointType>* BuildBoxingKernel(
    int32_t in_num, int32_t out_num, int kernel_name,
    BoxingOpConf::InBoxCase in_box_case,
    BoxingOpConf::OutBoxCase out_box_case) {
  // config boxing operator from box cases
  OperatorConf op_conf;
  op_conf.set_name("boxing_test" + std::to_string(kernel_name));
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
  auto boxing_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  boxing_op->ToProto(&op_proto);
  auto boxing_kernel = new BoxingKernel<DeviceType::kCPU, FloatingPointType>;
  boxing_kernel->InitFromOpProto(op_proto);
  return boxing_kernel;
}

template<typename FloatingPointType>
std::function<Blob*(const std::string&)> ConstructBnInOp2BlobPtr(
    const std::vector<std::vector<int64_t>>& in_dim_vecs,
    const std::vector<std::vector<int64_t>>& out_dim_vecs,
    const std::vector<int64_t> middle_dim = {0, 0, 0, 0}) {
  int32_t in_num = in_dim_vecs.size();
  int32_t out_num = out_dim_vecs.size();
  using KTCommon = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
  // construct mapping from bns to blobs
  auto bn2blob_ptr = new std::map<std::string, Blob*>;
  for (size_t i = 0; i < in_num; ++i) {
    bn2blob_ptr->emplace(
        "in_" + std::to_string(i),
        KTCommon::CreateBlobWithSameValue(in_dim_vecs[i], (i + 1) * 1.0));
    bn2blob_ptr->emplace("in_" + std::to_string(i) + "_diff",
                         KTCommon::CreateBlobWithSameValue(in_dim_vecs[i], 0));
  }
  for (size_t i = 0; i < out_num; ++i) {
    bn2blob_ptr->emplace(
        "out_" + std::to_string(i),
        KTCommon::CreateBlobWithSameValue(out_dim_vecs[i], (i + 1) * 10.0));
    bn2blob_ptr->emplace(
        "out_" + std::to_string(i) + "_diff",
        KTCommon::CreateBlobWithSameValue(out_dim_vecs[i], (i + 1) * 1.0));
  }
  bn2blob_ptr->emplace(std::string("middle"),
                       KTCommon::CreateBlobWithSameValue(middle_dim, 0));

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

// This version use the output blobs in BnInOp2BlobPtr as the input blobs
template<typename FloatingPointType>
std::function<Blob*(const std::string&)> ConstructBnInOp2BlobPtr(
    std::function<Blob*(const std::string&)> bn2bptr,
    const std::vector<std::vector<int64_t>>& in_dim_vecs,
    const std::vector<std::vector<int64_t>>& out_dim_vecs,
    const std::vector<int64_t> middle_dim = {0, 0, 0, 0}) {
  using KTCommon = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
  auto bn2blob_ptr = new std::map<std::string, Blob*>;
  for (size_t i = 0; i < out_dim_vecs.size(); ++i) {
    Blob* b = bn2bptr("out_" + std::to_string(i));
    bn2blob_ptr->emplace("in_" + std::to_string(i), b);

    b = bn2bptr("out_" + std::to_string(i) + "_diff");
    bn2blob_ptr->emplace("in_" + std::to_string(i) + "_diff", b);
  }

  bn2blob_ptr->emplace(std::string("middle"),
                       KTCommon::CreateBlobWithSameValue(middle_dim, 0));

  for (size_t i = 0; i < in_dim_vecs.size(); ++i) {
    bn2blob_ptr->emplace("out_" + std::to_string(i),
                         KTCommon::CreateBlobWithSameValue(in_dim_vecs[i], 0));
    bn2blob_ptr->emplace(
        "out_" + std::to_string(i) + "_diff",
        KTCommon::CreateBlobWithSameValue(in_dim_vecs[i], (i + 1) * 10.0));
  }

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<typename FloatingPointType>
void TestBoxingKernelConcatClone() {
  using KTCommon = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  auto boxing_kernel_0 = BuildBoxingKernel<FloatingPointType>(
      4, 1, 0, BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);
  auto boxing_kernel_1 = BuildBoxingKernel<FloatingPointType>(
      4, 5, 1, BoxingOpConf::kConcatBox, BoxingOpConf::kCloneBox);

  // Build mapping bns->blobs with first kernel
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {3, 4, 5, 5}, {3, 2, 5, 5}, {3, 1, 5, 5}, {3, 7, 5, 5}};
  std::vector<std::vector<int64_t>> out_dim_vecs_0 = {{3, 14, 5, 5}};
  std::vector<std::vector<int64_t>> out_dim_vecs_1 = {{3, 14, 5, 5},
                                                      {3, 14, 5, 5},
                                                      {3, 14, 5, 5},
                                                      {3, 14, 5, 5},
                                                      {3, 14, 5, 5}};
  auto BnInOp2BlobPtr_0 = ConstructBnInOp2BlobPtr<FloatingPointType>(
      in_dim_vecs, out_dim_vecs_0, out_dim_vecs_0[0]);

  auto BnInOp2BlobPtr_1 = ConstructBnInOp2BlobPtr<FloatingPointType>(
      in_dim_vecs, out_dim_vecs_1, out_dim_vecs_0[0]);

  // Run forward && backward test
  boxing_kernel_0->Forward(ctx, BnInOp2BlobPtr_0);
  boxing_kernel_1->Forward(ctx, BnInOp2BlobPtr_1);
  boxing_kernel_1->Backward(ctx, BnInOp2BlobPtr_1);
  boxing_kernel_0->Backward(ctx, BnInOp2BlobPtr_0);

  KTCommon::SyncStream(&ctx);

  for (size_t i = 0; i < in_dim_vecs.size(); ++i) {
    Blob* expected_in_diff =
        KTCommon::CreateBlobWithSameValue(in_dim_vecs[i], 15.0);
    Blob* in_i_diff = BnInOp2BlobPtr_1("in_" + std::to_string(i) + "_diff");
    KTCommon::BlobCmp(in_i_diff, expected_in_diff);
  }

  Blob* out_0 = BnInOp2BlobPtr_0("out_0");
  for (size_t i = 1; i < out_dim_vecs_1.size(); ++i) {
    Blob* out_i = BnInOp2BlobPtr_1("out_" + std::to_string(i));
    KTCommon::BlobCmp(out_i, out_0);
  }
}

template<typename FloatingPointType>
void TestBoxingKernelConcatSplit() {
  // Create cpu_device and kernel contexts
  using KTCommon = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;

  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  // Build boxing kernels
  auto boxing_kernel_0 = BuildBoxingKernel<FloatingPointType>(
      4, 3, 0, BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);
  auto boxing_kernel_1 = BuildBoxingKernel<FloatingPointType>(
      3, 4, 1, BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);

  // Build blobs
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {3, 4, 5, 5}, {3, 2, 5, 5}, {3, 1, 5, 5}, {3, 7, 5, 5}};
  std::vector<std::vector<int64_t>> out_dim_vecs = {
      {3, 5, 5, 5}, {3, 6, 5, 5}, {3, 3, 5, 5}};
  auto BnInOp2BlobPtr_0 = ConstructBnInOp2BlobPtr<FloatingPointType>(
      in_dim_vecs, out_dim_vecs, {3, 14, 5, 5});

  auto BnInOp2BlobPtr_1 = ConstructBnInOp2BlobPtr<FloatingPointType>(
      BnInOp2BlobPtr_0, in_dim_vecs, out_dim_vecs, {3, 14, 5, 5});

  // Run forward && backward test
  boxing_kernel_0->Forward(ctx, BnInOp2BlobPtr_0);
  boxing_kernel_1->Forward(ctx, BnInOp2BlobPtr_1);
  boxing_kernel_1->Backward(ctx, BnInOp2BlobPtr_1);
  boxing_kernel_0->Backward(ctx, BnInOp2BlobPtr_0);

  KTCommon::SyncStream(&ctx);

  // Check input && output blobs in this graph should be the same
  for (size_t i = 0; i < in_dim_vecs.size(); ++i) {
    KTCommon::BlobCmp(BnInOp2BlobPtr_0("in_" + std::to_string(i)),
                      BnInOp2BlobPtr_1("out_" + std::to_string(i)));
    KTCommon::BlobCmp(BnInOp2BlobPtr_0("in_" + std::to_string(i) + "_diff"),
                      BnInOp2BlobPtr_1("out_" + std::to_string(i) + "_diff"));
  }
}

template<typename FloatingPointType>
void TestBoxingKernelConcatSplitNull() {
  // Create cpu_device and kernel contexts
  using KTCommon = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  // Build boxing kernels
  auto boxing_kernel_0 = BuildBoxingKernel<FloatingPointType>(
      4, 3, 0, BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);
  auto boxing_kernel_1 = BuildBoxingKernel<FloatingPointType>(
      3, 4, 1, BoxingOpConf::kConcatBox, BoxingOpConf::kDataSplitBox);

  // Build blobs
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {3, 4, 5, 5}, {3, 2, 5, 5}, {3, 1, 5, 5}, {3, 7, 5, 5}, {3, 0, 5, 5}};
  std::vector<std::vector<int64_t>> out_dim_vecs = {
      {3, 5, 5, 5}, {3, 6, 5, 5}, {3, 3, 5, 5}, {3, 0, 5, 5}, {3, 0, 5, 5}};
  auto BnInOp2BlobPtr_0 = ConstructBnInOp2BlobPtr<FloatingPointType>(
      in_dim_vecs, out_dim_vecs, {3, 14, 5, 5});

  auto BnInOp2BlobPtr_1 = ConstructBnInOp2BlobPtr<FloatingPointType>(
      BnInOp2BlobPtr_0, in_dim_vecs, out_dim_vecs, {3, 14, 5, 5});

  // Run forward && backward test
  boxing_kernel_0->Forward(ctx, BnInOp2BlobPtr_0);
  boxing_kernel_1->Forward(ctx, BnInOp2BlobPtr_1);
  boxing_kernel_1->Backward(ctx, BnInOp2BlobPtr_1);
  boxing_kernel_0->Backward(ctx, BnInOp2BlobPtr_0);

  KTCommon::SyncStream(&ctx);

  // Check: input && output blobs in this graph should be the same
  for (size_t i = 0; i < in_dim_vecs.size(); ++i) {
    KTCommon::BlobCmp(BnInOp2BlobPtr_0("in_" + std::to_string(i)),
                      BnInOp2BlobPtr_1("out_" + std::to_string(i)));
    KTCommon::BlobCmp(BnInOp2BlobPtr_0("in_" + std::to_string(i) + "_diff"),
                      BnInOp2BlobPtr_1("out_" + std::to_string(i) + "_diff"));
  }
}

template<typename FloatingPointType>
void TestBoxingKernelAddClone() {
  // Create cpu_device and kernel contexts
  KernelCtx ctx;
  using KTCommon = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;
  KTCommon::BuildKernelCtx(&ctx);

  // Build boxing kernel
  auto boxing_kernel = BuildBoxingKernel<FloatingPointType>(
      4, 3, 0, BoxingOpConf::kAddBox, BoxingOpConf::kCloneBox);

  // Build mapping bns->blobs
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {3, 4, 5, 5}, {3, 4, 5, 5}, {3, 4, 5, 5}, {3, 4, 5, 5}};
  std::vector<std::vector<int64_t>> out_dim_vecs = {
      {3, 4, 5, 5}, {3, 4, 5, 5}, {3, 4, 5, 5}};
  auto BnInOp2BlobPtr = ConstructBnInOp2BlobPtr<FloatingPointType>(
      in_dim_vecs, out_dim_vecs, {3, 4, 5, 5});

  // Run forward && backward
  boxing_kernel->Forward(ctx, BnInOp2BlobPtr);

  KTCommon::SyncStream(&ctx);

  // check if add-results is the same as expected.
  Blob* expected_add_b =
      KTCommon::CreateBlobWithSameValue(out_dim_vecs[0], 10.0);
  for (size_t i = 0; i < out_dim_vecs.size(); ++i) {
    KTCommon::BlobCmp(BnInOp2BlobPtr("out_" + std::to_string(i)),
                      expected_add_b);
  }
}

}  // namespace

}  // namespace test

TEST(boxingkernel, boxing_add_clone_box) {
  test::TestBoxingKernelAddClone<float>();
  test::TestBoxingKernelAddClone<double>();
}

TEST(boxingKernel, boxing_concat_clone_box) {
  test::TestBoxingKernelConcatClone<float>();
  test::TestBoxingKernelConcatClone<double>();
}

TEST(boxingKernel, boxing_concat_split_box) {
  test::TestBoxingKernelConcatSplit<float>();
  test::TestBoxingKernelConcatSplit<double>();
}

TEST(boxingKernel, boxing_concat_split_box_with_null) {
  test::TestBoxingKernelConcatSplitNull<float>();
  test::TestBoxingKernelConcatSplitNull<double>();
}

}  // namespace oneflow
