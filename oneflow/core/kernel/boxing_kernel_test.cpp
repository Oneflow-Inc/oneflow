#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<typename T>
BoxingKernel<T>* BuildBoxingKernel(int32_t in_num, int32_t out_num,
                                   int kernel_name,
                                   BoxingOpConf::InBoxCase in_box_case,
                                   BoxingOpConf::OutBoxCase out_box_case,
                                   int32_t concat_axis = 1) {
  // config boxing operator from box cases
  OperatorConf op_conf;
  op_conf.set_name("boxing_test" + std::to_string(kernel_name));
  BoxingOpConf* boxing_conf = op_conf.mutable_boxing_conf();
  boxing_conf->set_in_num(in_num);
  boxing_conf->set_out_num(out_num);
  if (in_box_case == BoxingOpConf::kConcatBox) {
    boxing_conf->mutable_concat_box()->set_axis(concat_axis);
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
  auto boxing_kernel = new BoxingKernel<T>;
  boxing_kernel->InitFromOpProto(op_proto);
  return boxing_kernel;
}

template<typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobMap(
    const std::vector<std::vector<int64_t>>& in_dim_vecs,
    const std::vector<std::vector<int64_t>>& out_dim_vecs,
    const std::vector<int64_t> middle_dim = {0, 0, 0, 0}) {
  DataType data_type = GetDataType<T>::val;
  int32_t in_num = in_dim_vecs.size();
  int32_t out_num = out_dim_vecs.size();
  using KTC = KTCommon<DeviceType::kCPU, T>;
  // construct mapping from bns to blobs
  auto bn2blob = new std::map<std::string, Blob*>;
  for (size_t i = 0; i < in_num; ++i) {
    bn2blob->emplace("in_" + std::to_string(i),
                     KTC::CreateBlobWithSameVal(
                         new BlobDesc(Shape(in_dim_vecs[i]), data_type, false),
                         (i + 1) * 1.0));
    bn2blob->emplace("in_" + std::to_string(i) + "_diff",
                     KTC::CreateBlobWithRandomVal(new BlobDesc(
                         Shape(in_dim_vecs[i]), data_type, false)));
  }
  for (size_t i = 0; i < out_num; ++i) {
    bn2blob->emplace("out_" + std::to_string(i),
                     KTC::CreateBlobWithSameVal(
                         new BlobDesc(Shape(out_dim_vecs[i]), data_type, false),
                         (i + 1) * 10.0));
    bn2blob->emplace("out_" + std::to_string(i) + "_diff",
                     KTC::CreateBlobWithSameVal(
                         new BlobDesc(Shape(out_dim_vecs[i]), data_type, false),
                         (i + 1) * 1.0));
  }
  bn2blob->emplace(std::string("middle"),
                   KTC::CreateBlobWithRandomVal(
                       new BlobDesc(Shape(middle_dim), data_type, false)));

  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

// This version use the output blobs in BnInOp2BlobMap as the input blobs
template<typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobMap(
    std::function<Blob*(const std::string&)> bn2bptr,
    const std::vector<std::vector<int64_t>>& in_dim_vecs,
    const std::vector<std::vector<int64_t>>& out_dim_vecs,
    const std::vector<int64_t> middle_dim = {0, 0, 0, 0}) {
  using KTC = KTCommon<DeviceType::kCPU, T>;
  DataType data_type = GetDataType<T>::val;
  auto bn2blob = new std::map<std::string, Blob*>;
  for (size_t i = 0; i < out_dim_vecs.size(); ++i) {
    Blob* b = bn2bptr("out_" + std::to_string(i));
    bn2blob->emplace("in_" + std::to_string(i), b);

    b = bn2bptr("out_" + std::to_string(i) + "_diff");
    bn2blob->emplace("in_" + std::to_string(i) + "_diff", b);
  }

  bn2blob->emplace(std::string("middle"),
                   KTC::CreateBlobWithRandomVal(
                       new BlobDesc(Shape(middle_dim), data_type, false)));

  for (size_t i = 0; i < in_dim_vecs.size(); ++i) {
    bn2blob->emplace("out_" + std::to_string(i),
                     KTC::CreateBlobWithRandomVal(new BlobDesc(
                         Shape(in_dim_vecs[i]), data_type, false)));
    bn2blob->emplace("out_" + std::to_string(i) + "_diff",
                     KTC::CreateBlobWithSameVal(
                         new BlobDesc(Shape(in_dim_vecs[i]), data_type, false),
                         (i + 1) * 10.0));
  }

  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<typename T>
void TestBoxingKernelConcatClone() {
  using KTC = KTCommon<DeviceType::kCPU, T>;
  DataType data_type = GetDataType<T>::val;
  KernelCtx ctx;
  BuildKernelCtx<DeviceType::kCPU>(&ctx);

  auto boxing_kernel_0 = BuildBoxingKernel<T>(4, 1, 0, BoxingOpConf::kConcatBox,
                                              BoxingOpConf::kDataSplitBox);
  auto boxing_kernel_1 = BuildBoxingKernel<T>(4, 5, 1, BoxingOpConf::kConcatBox,
                                              BoxingOpConf::kCloneBox);

  // Build mapping bns->blobs with first kernel
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {3, 4, 5, 5}, {3, 2, 5, 5}, {3, 1, 5, 5}, {3, 7, 5, 5}};
  std::vector<std::vector<int64_t>> out_dim_vecs_0 = {{3, 14, 5, 5}};
  std::vector<std::vector<int64_t>> out_dim_vecs_1 = {{3, 14, 5, 5},
                                                      {3, 14, 5, 5},
                                                      {3, 14, 5, 5},
                                                      {3, 14, 5, 5},
                                                      {3, 14, 5, 5}};
  auto BnInOp2BlobMap_0 =
      BuildBnInOp2BlobMap<T>(in_dim_vecs, out_dim_vecs_0, out_dim_vecs_0[0]);

  auto BnInOp2BlobMap_1 =
      BuildBnInOp2BlobMap<T>(in_dim_vecs, out_dim_vecs_1, out_dim_vecs_0[0]);

  // Run forward && backward test
  boxing_kernel_0->Forward(ctx, BnInOp2BlobMap_0);
  boxing_kernel_1->Forward(ctx, BnInOp2BlobMap_1);
  boxing_kernel_1->Backward(ctx, BnInOp2BlobMap_1);
  boxing_kernel_0->Backward(ctx, BnInOp2BlobMap_0);

  SyncStream<DeviceType::kCPU>(&ctx);

  for (size_t i = 0; i < in_dim_vecs.size(); ++i) {
    Blob* expected_in_diff = KTC::CreateBlobWithSameVal(
        new BlobDesc(Shape(in_dim_vecs[i]), data_type, false),
        static_cast<T>(15));
    Blob* in_i_diff = BnInOp2BlobMap_1("in_" + std::to_string(i) + "_diff");
    KTC::BlobCmp(in_i_diff, expected_in_diff);
  }

  Blob* out_0 = BnInOp2BlobMap_0("out_0");
  for (size_t i = 1; i < out_dim_vecs_1.size(); ++i) {
    Blob* out_i = BnInOp2BlobMap_1("out_" + std::to_string(i));
    KTC::BlobCmp(out_i, out_0);
  }
}

template<typename T>
void TestBoxingKernelConcatSplit_1() {
  // Create cpu_device and kernel contexts
  using KTC = KTCommon<DeviceType::kCPU, T>;

  DataType data_type = GetDataType<T>::val;
  KernelCtx ctx;
  BuildKernelCtx<DeviceType::kCPU>(&ctx);

  // Build boxing kernels
  auto boxing_kernel = BuildBoxingKernel<T>(4, 2, 0, BoxingOpConf::kConcatBox,
                                            BoxingOpConf::kDataSplitBox, 0);

  // Build blobs
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {1, 1, 2, 1}, {2, 1, 2, 1}, {1, 1, 2, 1}, {3, 1, 2, 1}};
  std::vector<std::vector<int64_t>> out_dim_vecs = {{3, 1, 2, 1}, {4, 1, 2, 1}};

  auto BnInOp2BlobMap =
      BuildBnInOp2BlobMap<T>(in_dim_vecs, out_dim_vecs, {7, 1, 2, 1});

  // Run forward && backward test
  boxing_kernel->Forward(ctx, BnInOp2BlobMap);
  boxing_kernel->Backward(ctx, BnInOp2BlobMap);

  SyncStream<DeviceType::kCPU>(&ctx);

  // Check input && output blobs in this graph should be the same
  T ex_out_0_mat[] = {1, 1, 2, 2, 2, 2};
  T ex_out_1_mat[] = {3, 3, 4, 4, 4, 4, 4, 4};
  Blob* expected_out_0 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 1, 2, 1}), data_type, false), ex_out_0_mat);
  Blob* expected_out_1 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({4, 1, 2, 1}), data_type, false), ex_out_1_mat);
  KTC::BlobCmp(BnInOp2BlobMap("out_0"), expected_out_0);
  KTC::BlobCmp(BnInOp2BlobMap("out_1"), expected_out_1);

  T ex_in_diff_0_mat[] = {1, 1};
  T ex_in_diff_1_mat[] = {1, 1, 1, 1};
  T ex_in_diff_2_mat[] = {2, 2};
  T ex_in_diff_3_mat[] = {2, 2, 2, 2, 2, 2};
  Blob* expected_in_diff_0 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 1, 2, 1}), data_type, false), ex_in_diff_0_mat);
  Blob* expected_in_diff_1 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 1, 2, 1}), data_type, false), ex_in_diff_1_mat);
  Blob* expected_in_diff_2 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 1, 2, 1}), data_type, false), ex_in_diff_2_mat);
  Blob* expected_in_diff_3 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 1, 2, 1}), data_type, false), ex_in_diff_3_mat);
  KTC::BlobCmp(BnInOp2BlobMap("in_0_diff"), expected_in_diff_0);
  KTC::BlobCmp(BnInOp2BlobMap("in_1_diff"), expected_in_diff_1);
  KTC::BlobCmp(BnInOp2BlobMap("in_2_diff"), expected_in_diff_2);
  KTC::BlobCmp(BnInOp2BlobMap("in_3_diff"), expected_in_diff_3);
}

template<typename T>
void TestBoxingKernelConcatSplit() {
  // Create cpu_device and kernel contexts
  using KTC = KTCommon<DeviceType::kCPU, T>;

  DataType data_type = GetDataType<T>::val;
  KernelCtx ctx;
  BuildKernelCtx<DeviceType::kCPU>(&ctx);

  // Build boxing kernels
  auto boxing_kernel = BuildBoxingKernel<T>(4, 2, 0, BoxingOpConf::kConcatBox,
                                            BoxingOpConf::kDataSplitBox);

  // Build blobs
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {3, 1, 2, 1}, {3, 2, 2, 1}, {3, 3, 2, 1}, {3, 4, 2, 1}};
  std::vector<std::vector<int64_t>> out_dim_vecs = {{2, 10, 2, 1},
                                                    {1, 10, 2, 1}};
  auto BnInOp2BlobMap =
      BuildBnInOp2BlobMap<T>(in_dim_vecs, out_dim_vecs, {3, 10, 2, 1});

  // Run forward && backward test
  boxing_kernel->Forward(ctx, BnInOp2BlobMap);
  boxing_kernel->Backward(ctx, BnInOp2BlobMap);

  SyncStream<DeviceType::kCPU>(&ctx);

  // Check input && output blobs in this graph should be the same
  T ex_out_0_mat[] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4,
                      4, 4, 4, 4, 4, 4, 1, 1, 2, 2, 2, 2, 3, 3,
                      3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  T ex_out_1_mat[] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  Blob* expected_out_0 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 10, 2, 1}), data_type, false), ex_out_0_mat);
  Blob* expected_out_1 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 10, 2, 1}), data_type, false), ex_out_1_mat);
  KTC::BlobCmp(BnInOp2BlobMap("out_0"), expected_out_0);
  KTC::BlobCmp(BnInOp2BlobMap("out_1"), expected_out_1);

  T ex_in_diff_0_mat[] = {1, 1, 1, 1, 2, 2};
  T ex_in_diff_1_mat[] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2};
  T ex_in_diff_2_mat[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
  T ex_in_diff_3_mat[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
  Blob* expected_in_diff_0 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 1, 2, 1}), data_type, false), ex_in_diff_0_mat);
  Blob* expected_in_diff_1 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 2, 2, 1}), data_type, false), ex_in_diff_1_mat);
  Blob* expected_in_diff_2 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 3, 2, 1}), data_type, false), ex_in_diff_2_mat);
  Blob* expected_in_diff_3 = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 4, 2, 1}), data_type, false), ex_in_diff_3_mat);
  KTC::BlobCmp(BnInOp2BlobMap("in_0_diff"), expected_in_diff_0);
  KTC::BlobCmp(BnInOp2BlobMap("in_1_diff"), expected_in_diff_1);
  KTC::BlobCmp(BnInOp2BlobMap("in_2_diff"), expected_in_diff_2);
  KTC::BlobCmp(BnInOp2BlobMap("in_3_diff"), expected_in_diff_3);
}

template<typename T>
void TestBoxingKernelConcatSplitNull() {
  // Create cpu_device and kernel contexts
  using KTCommon = KTCommon<DeviceType::kCPU, T>;

  DataType data_type = GetDataType<T>::val;
  KernelCtx ctx;
  BuildKernelCtx<DeviceType::kCPU>(&ctx);

  // Build boxing kernels
  auto boxing_kernel = BuildBoxingKernel<T>(4, 2, 0, BoxingOpConf::kConcatBox,
                                            BoxingOpConf::kDataSplitBox);

  // Build blobs
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {3, 1, 2, 1}, {3, 2, 2, 1}, {3, 3, 2, 1}, {3, 4, 2, 1}, {3, 0, 2, 1}};
  std::vector<std::vector<int64_t>> out_dim_vecs = {
      {2, 10, 2, 1}, {1, 10, 2, 1}, {2, 0, 3, 1}};
  auto BnInOp2BlobMap =
      BuildBnInOp2BlobMap<T>(in_dim_vecs, out_dim_vecs, {3, 10, 2, 1});

  // Run forward && backward test
  boxing_kernel->Forward(ctx, BnInOp2BlobMap);
  boxing_kernel->Backward(ctx, BnInOp2BlobMap);

  SyncStream<DeviceType::kCPU>(&ctx);

  // Check input && output blobs in this graph should be the same
  T ex_out_0_mat[] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4,
                      4, 4, 4, 4, 4, 4, 1, 1, 2, 2, 2, 2, 3, 3,
                      3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  T ex_out_1_mat[] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                      3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  Blob* expected_out_0 = KTCommon::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 10, 2, 1}), data_type, false), ex_out_0_mat);
  Blob* expected_out_1 = KTCommon::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 10, 2, 1}), data_type, false), ex_out_1_mat);
  KTCommon::BlobCmp(BnInOp2BlobMap("out_0"), expected_out_0);
  KTCommon::BlobCmp(BnInOp2BlobMap("out_1"), expected_out_1);

  T ex_in_diff_0_mat[] = {1, 1, 1, 1, 2, 2};
  T ex_in_diff_1_mat[] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2};
  T ex_in_diff_2_mat[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
  T ex_in_diff_3_mat[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
  Blob* expected_in_diff_0 = KTCommon::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 1, 2, 1}), data_type, false), ex_in_diff_0_mat);
  Blob* expected_in_diff_1 = KTCommon::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 2, 2, 1}), data_type, false), ex_in_diff_1_mat);
  Blob* expected_in_diff_2 = KTCommon::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 3, 2, 1}), data_type, false), ex_in_diff_2_mat);
  Blob* expected_in_diff_3 = KTCommon::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({3, 4, 2, 1}), data_type, false), ex_in_diff_3_mat);
  KTCommon::BlobCmp(BnInOp2BlobMap("in_0_diff"), expected_in_diff_0);
  KTCommon::BlobCmp(BnInOp2BlobMap("in_1_diff"), expected_in_diff_1);
  KTCommon::BlobCmp(BnInOp2BlobMap("in_2_diff"), expected_in_diff_2);
  KTCommon::BlobCmp(BnInOp2BlobMap("in_3_diff"), expected_in_diff_3);
}

template<typename T>
void TestBoxingKernelAddClone() {
  // Create cpu_device and kernel contexts
  KernelCtx ctx;
  using KTCommon = KTCommon<DeviceType::kCPU, T>;
  BuildKernelCtx<DeviceType::kCPU>(&ctx);

  // Build boxing kernel
  auto boxing_kernel = BuildBoxingKernel<T>(4, 3, 0, BoxingOpConf::kAddBox,
                                            BoxingOpConf::kCloneBox);

  // Build mapping bns->blobs
  std::vector<std::vector<int64_t>> in_dim_vecs = {
      {3, 4, 5, 5}, {3, 4, 5, 5}, {3, 4, 5, 5}, {3, 4, 5, 5}};
  std::vector<std::vector<int64_t>> out_dim_vecs = {
      {3, 4, 5, 5}, {3, 4, 5, 5}, {3, 4, 5, 5}};
  auto BnInOp2BlobMap =
      BuildBnInOp2BlobMap<T>(in_dim_vecs, out_dim_vecs, {3, 4, 5, 5});

  // Run forward && backward
  boxing_kernel->Forward(ctx, BnInOp2BlobMap);

  SyncStream<DeviceType::kCPU>(&ctx);

  // check if add-results is the same as expected.
  Blob* expected_add_b = KTCommon::CreateBlobWithSameVal(
      new BlobDesc(Shape(out_dim_vecs[0]), GetDataType<T>::val, false), 10.0);
  for (size_t i = 0; i < out_dim_vecs.size(); ++i) {
    KTCommon::BlobCmp(BnInOp2BlobMap("out_" + std::to_string(i)),
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
  test::TestBoxingKernelConcatSplit_1<double>();
  test::TestBoxingKernelConcatSplit_1<double>();
}

TEST(boxingKernel, boxing_concat_split_box_with_null) {
  test::TestBoxingKernelConcatSplitNull<float>();
  test::TestBoxingKernelConcatSplitNull<double>();
}

}  // namespace oneflow
