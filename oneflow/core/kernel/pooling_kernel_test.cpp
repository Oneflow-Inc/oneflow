#include "oneflow/core/kernel/pooling_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildPoolingKernel(const PoolingOpConf_PoolMethod& pooling_method) {
  OperatorConf op_conf;
  op_conf.set_name("pooling_test");
  PoolingOpConf* pooling_conf = op_conf.mutable_pooling_conf();
  pooling_conf->set_in("pooling_in");
  pooling_conf->set_out("pooling_out");
  pooling_conf->set_pool(pooling_method);
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
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
    const PoolingOpConf_PoolMethod& pooling_method) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  FloatingPointType in_mat[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                10, 11, 12, 13, 14, 15, 16, 17, 18,
                                19, 20, 21, 22, 23, 24, 25};
  FloatingPointType expected_max_out_mat[] = {7, 9, 10, 17, 19, 20, 22, 24, 25};
  FloatingPointType expected_max_in_diff_mat[] = {
      0, 0, 0, 0,  0, 0,  7,  0, 9,  10, 0,  0, 0,
      0, 0, 0, 17, 0, 19, 20, 0, 22, 0,  24, 25};
  FloatingPointType expected_ave_out_mat[] = {16.0 / 9, 33.0 / 9,  28.0 / 9,
                                              69.0 / 9, 13,        87.0 / 9,
                                              76.0 / 9, 123.0 / 9, 88.0 / 9};
  FloatingPointType expected_ave_in_diff_mat[] = {
      16.0 / 9 / 9,
      16.0 / 9 / 9 + 33.0 / 9 / 9,
      33.0 / 9 / 9,
      33.0 / 9 / 9 + 28.0 / 9 / 9,
      28.0 / 9 / 9,
      16.0 / 9 / 9 + 69.0 / 9 / 9,
      16.0 / 9 / 9 + 33.0 / 9 / 9 + 69.0 / 9 / 9 + 13.0 / 9,
      33.0 / 9 / 9 + 13.0 / 9,
      33.0 / 9 / 9 + 28.0 / 9 / 9 + 13.0 / 9 + 87.0 / 9 / 9,
      28.0 / 9 / 9 + 87.0 / 9 / 9,
      69.0 / 9 / 9,
      69.0 / 9 / 9 + 13.0 / 9,
      13.0 / 9,
      13.0 / 9 + 87.0 / 9 / 9,
      87.0 / 9 / 9,
      69.0 / 9 / 9 + 76.0 / 9 / 9,
      69.0 / 9 / 9 + 13.0 / 9 + 76.0 / 9 / 9 + 123.0 / 9 / 9,
      13.0 / 9 + 123.0 / 9 / 9,
      13.0 / 9 + 87.0 / 9 / 9 + 123.0 / 9 / 9 + 88.0 / 9 / 9,
      87.0 / 9 / 9 + 88.0 / 9 / 9,
      76.0 / 9 / 9,
      76.0 / 9 / 9 + 123.0 / 9 / 9,
      123.0 / 9 / 9,
      123.0 / 9 / 9 + 88.0 / 9 / 9,
      88.0 / 9 / 9};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({1, 1, 5, 5}, in_mat);
  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithSameValue({1, 1, 3, 3}, 0);
  (*bn2blob_ptr)["idx"] = KTCommon::CreateBlobWithSameValue({1, 1, 5, 5}, 0);
  (*bn2blob_ptr)["out_diff"] = (*bn2blob_ptr)["out"];
  (*bn2blob_ptr)["in_diff"] =
      KTCommon::CreateBlobWithSameValue({1, 1, 5, 5}, 0);
  if (pooling_method == PoolingOpConf::MAX) {
    (*bn2blob_ptr)["expected_out"] =
        KTCommon::CreateBlobWithVector({1, 1, 3, 3}, expected_max_out_mat);
    (*bn2blob_ptr)["expected_in_diff"] =
        KTCommon::CreateBlobWithVector({1, 1, 5, 5}, expected_max_in_diff_mat);
  } else if (pooling_method == PoolingOpConf::AVE) {
    (*bn2blob_ptr)["expected_out"] =
        KTCommon::CreateBlobWithVector({1, 1, 3, 3}, expected_ave_out_mat);
    (*bn2blob_ptr)["expected_in_diff"] =
        KTCommon::CreateBlobWithVector({1, 1, 5, 5}, expected_ave_in_diff_mat);
  } else {
    TODO();
  }

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
void TestPoolingKernel(const PoolingOpConf_PoolMethod& pooling_method) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  auto BnInOp2BlobPtr =
      BuildBnInOp2BlobPtr<device_type, FloatingPointType>(pooling_method);
  auto pooling_kernel =
      BuildPoolingKernel<device_type, FloatingPointType>(pooling_method);

  pooling_kernel->Forward(ctx, BnInOp2BlobPtr);
  pooling_kernel->Backward(ctx, BnInOp2BlobPtr);
  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
}

}  // namespace

}  // namespace test

TEST(PoolingKernel, pooling_max_cpu) {
  test::TestPoolingKernel<DeviceType::kCPU, float>(PoolingOpConf::MAX);
  test::TestPoolingKernel<DeviceType::kCPU, double>(PoolingOpConf::MAX);
}

TEST(PoolingKernel, pooling_ave_cpu) {
  test::TestPoolingKernel<DeviceType::kCPU, float>(PoolingOpConf::AVE);
  test::TestPoolingKernel<DeviceType::kCPU, double>(PoolingOpConf::AVE);
}

TEST(PoolingKernel, pooling_max_gpu) {
  test::TestPoolingKernel<DeviceType::kGPU, float>(PoolingOpConf::MAX);
  test::TestPoolingKernel<DeviceType::kGPU, double>(PoolingOpConf::MAX);
}

TEST(PoolingKernel, pooling_ave_gpu) {
  test::TestPoolingKernel<DeviceType::kGPU, float>(PoolingOpConf::AVE);
  test::TestPoolingKernel<DeviceType::kGPU, double>(PoolingOpConf::AVE);
}

}  // namespace oneflow
