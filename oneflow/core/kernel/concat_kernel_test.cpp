#include "oneflow/core/kernel/concat_kernel.h"
#include <random>
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildConcatKernel() {
  OperatorConf op_conf;
  op_conf.set_name("concat_test");
  ConcatOpConf* concat_conf = op_conf.mutable_concat_conf();
  concat_conf->add_in("concat/in0");
  concat_conf->add_in("concat/in1");
  concat_conf->set_axis(1);
  concat_conf->set_out("concat_kernel_test");
  auto concat_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  concat_op->ToProto(&op_proto);
  auto concat_kernel = new ConcatKernel<device_type, FloatingPointType>();
  concat_kernel->InitFromOpProto(op_proto);
  return concat_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  FloatingPointType in_0_mat[] = {1, 2, 3, 4, 5, 6};
  FloatingPointType in_1_mat[] = {7, 8, 9, 10, 11, 12};
  FloatingPointType out_mat[12] = {0};
  FloatingPointType expected_out_mat[] = {1, 2, 3, 7,  8,  9,
                                          4, 5, 6, 10, 11, 12};
  FloatingPointType out_diff_mat[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  FloatingPointType in_0_diff_mat[6] = {0};
  FloatingPointType in_1_diff_mat[6] = {0};
  FloatingPointType expected_in_0_diff_mat[] = {1, 2, 3, 7, 8, 9};
  FloatingPointType expected_in_1_diff_mat[] = {4, 5, 6, 10, 11, 12};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  (*bn2blob_ptr)["in_0"] = KTCommon::CreateBlobWithVector({2, 3}, in_0_mat);
  (*bn2blob_ptr)["in_1"] = KTCommon::CreateBlobWithVector({2, 3}, in_1_mat);
  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithVector({2, 6}, out_mat);
  (*bn2blob_ptr)["expected_out"] =
      KTCommon::CreateBlobWithVector({2, 6}, expected_out_mat);
  (*bn2blob_ptr)["out_diff"] =
      KTCommon::CreateBlobWithVector({2, 6}, out_diff_mat);
  (*bn2blob_ptr)["in_0_diff"] =
      KTCommon::CreateBlobWithVector({2, 3}, in_0_diff_mat);
  (*bn2blob_ptr)["in_1_diff"] =
      KTCommon::CreateBlobWithVector({2, 3}, in_1_diff_mat);
  (*bn2blob_ptr)["expected_in_0_diff"] =
      KTCommon::CreateBlobWithVector({2, 3}, expected_in_0_diff_mat);
  (*bn2blob_ptr)["expected_in_1_diff"] =
      KTCommon::CreateBlobWithVector({2, 3}, expected_in_1_diff_mat);

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
void TestConcatKernel() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
  auto concat_kernel = BuildConcatKernel<device_type, FloatingPointType>();

  concat_kernel->Forward(ctx, BnInOp2BlobPtr);
  concat_kernel->Backward(ctx, BnInOp2BlobPtr);
  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_0_diff", "expected_in_0_diff");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_1_diff", "expected_in_1_diff");
}

}  // namespace

}  // namespace test

TEST(ConcatKernel, concat_cpu) {
  test::TestConcatKernel<DeviceType::kCPU, float>();
  test::TestConcatKernel<DeviceType::kCPU, double>();
}

TEST(ConcatKernel, concat_gpu) {
  test::TestConcatKernel<DeviceType::kGPU, float>();
  test::TestConcatKernel<DeviceType::kGPU, double>();
}  // namespace oneflow

}  // namespace oneflow
