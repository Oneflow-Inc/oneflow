#include "oneflow/core/kernel/pooling_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildPoolingKernel() {
  OperatorConf op_conf;
  op_conf.set_name("pooling_test");
  PoolingOpConf* pooling_conf = op_conf.mutable_pooling_conf();
  pooling_conf->set_in("pooling_in");
  pooling_conf->set_out("pooling_out");
  pooling_conf->set_pool(PoolingOpConf::MAX);
  pooling_conf->add_pad(1);
  pooling_conf->add_pad(1);
  pooling_conf->add_kernel_size(3);
  pooling_conf->add_kernel_size(3);
  pooling_conf->add_stride(2);
  pooling_conf->add_stride(2);

  auto pooling_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  pooling_op->ToProto(&op_proto);
  auto pooling_kernel = new PoolingKernel<device_type, FloatingPointType>();
  pooling_kernel->InitFromOpProto(op_proto);
  return pooling_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  FloatingPointType in_mat[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                10, 11, 12, 13, 14, 15, 16, 17, 18,
                                19, 20, 21, 22, 23, 24, 25};
  FloatingPointType out_mat[9] = {0};
  FloatingPointType index_mat[25] = {0};
  FloatingPointType out_diff_mat[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  FloatingPointType in_diff_mat[25] = {0};
  FloatingPointType expected_out_mat[] = {7, 9, 10, 17, 19, 20, 22, 24, 25};
  FloatingPointType expected_in_diff_mat[] = {0, 0, 0, 0, 0, 0, 9, 0, 8,
                                              7, 0, 0, 0, 0, 0, 0, 6, 0,
                                              5, 4, 0, 3, 0, 2, 1};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({1, 1, 5, 5}, in_mat);
  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithVector({1, 1, 3, 3}, out_mat);
  (*bn2blob_ptr)["idx"] =
      KTCommon::CreateBlobWithVector({1, 1, 5, 5}, index_mat);
  (*bn2blob_ptr)["out_diff"] =
      KTCommon::CreateBlobWithVector({1, 1, 3, 3}, out_diff_mat);
  (*bn2blob_ptr)["in_diff"] =
      KTCommon::CreateBlobWithVector({1, 1, 5, 5}, in_diff_mat);
  (*bn2blob_ptr)["expected_out"] =
      KTCommon::CreateBlobWithVector({1, 1, 3, 3}, expected_out_mat);
  (*bn2blob_ptr)["expected_in_diff"] =
      KTCommon::CreateBlobWithVector({1, 1, 5, 5}, expected_in_diff_mat);

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
void TestPoolingKernel() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
  auto pooling_kernel = BuildPoolingKernel<device_type, FloatingPointType>();

  pooling_kernel->Forward(ctx, BnInOp2BlobPtr);
  pooling_kernel->Backward(ctx, BnInOp2BlobPtr);
  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
}

}  // namespace

}  // namespace test

TEST(PoolingKernel, pooling_cpu) {
  test::TestPoolingKernel<DeviceType::kCPU, float>();
  test::TestPoolingKernel<DeviceType::kCPU, double>();
}

TEST(PoolingKernel, pooling_gpu) {
  test::TestPoolingKernel<DeviceType::kGPU, float>();
  test::TestPoolingKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
