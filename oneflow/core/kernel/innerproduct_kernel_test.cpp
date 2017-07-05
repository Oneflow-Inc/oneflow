#include "oneflow/core/kernel/innerproduct_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
    bool has_bias_term) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  FloatingPointType in_mat[] = {1, 2, 3, 4, 5, 6, 7, 8};
  FloatingPointType weight_mat[] = {5, 4, 5, 3, 2, 1, 7, 0, 1, 1, 9, 8};
  FloatingPointType bias_mat[] = {2, 3, 5};
  FloatingPointType bias_multiplier_mat[] = {1, 1};
  FloatingPointType out_mat[6] = {0};
  FloatingPointType in_diff_mat[8] = {0};
  FloatingPointType weight_diff_mat[12] = {0};
  FloatingPointType bias_diff_mat[3] = {0};

  FloatingPointType expected_out_without_bias_mat[] = {40,  25, 62,
                                                       108, 65, 138};
  FloatingPointType expected_in_diff_without_bias_mat[] = {
      312, 247, 933, 616, 808, 635, 2237, 1428};
  FloatingPointType expected_weight_diff_without_bias_mat[] = {
      580, 728, 876, 1024, 350, 440, 530, 620, 752, 952, 1152, 1352};
  FloatingPointType expected_out_mat[] = {42, 28, 67, 110, 68, 143};
  FloatingPointType expected_in_diff_mat[] = {333, 263, 1009, 662,
                                              829, 651, 2313, 1474};
  FloatingPointType expected_weight_diff_mat[] = {
      592, 744, 896, 1048, 368, 464, 560, 656, 782, 992, 1202, 1412};
  FloatingPointType expected_bias_diff_mat[] = {152, 96, 210};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({2, 4}, in_mat);
  (*bn2blob_ptr)["weight"] = KTCommon::CreateBlobWithVector({3, 4}, weight_mat);
  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithVector({2, 3}, out_mat);
  (*bn2blob_ptr)["out_diff"] = (*bn2blob_ptr)["out"];
  (*bn2blob_ptr)["in_diff"] =
      KTCommon::CreateBlobWithVector({2, 4}, in_diff_mat);
  (*bn2blob_ptr)["weight_diff"] =
      KTCommon::CreateBlobWithVector({3, 4}, weight_diff_mat);

  if (has_bias_term) {
    (*bn2blob_ptr)["bias"] = KTCommon::CreateBlobWithVector({1, 3}, bias_mat);
    (*bn2blob_ptr)["bias_multiplier"] =
        KTCommon::CreateBlobWithVector({2, 1}, bias_multiplier_mat);
    (*bn2blob_ptr)["bias_diff"] =
        KTCommon::CreateBlobWithVector({1, 3}, bias_diff_mat);
    (*bn2blob_ptr)["expected_bias_diff"] =
        KTCommon::CreateBlobWithVector({1, 3}, expected_bias_diff_mat);
    (*bn2blob_ptr)["expected_out"] =
        KTCommon::CreateBlobWithVector({2, 3}, expected_out_mat);
    (*bn2blob_ptr)["expected_in_diff"] =
        KTCommon::CreateBlobWithVector({2, 4}, expected_in_diff_mat);
    (*bn2blob_ptr)["expected_weight_diff"] =
        KTCommon::CreateBlobWithVector({3, 4}, expected_weight_diff_mat);
  } else {
    (*bn2blob_ptr)["expected_out"] =
        KTCommon::CreateBlobWithVector({2, 3}, expected_out_without_bias_mat);
    (*bn2blob_ptr)["expected_in_diff"] = KTCommon::CreateBlobWithVector(
        {2, 4}, expected_in_diff_without_bias_mat);
    (*bn2blob_ptr)["expected_weight_diff"] = KTCommon::CreateBlobWithVector(
        {3, 4}, expected_weight_diff_without_bias_mat);
  }
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildInnerProductKernel(bool has_bias_term) {
  OperatorConf op_conf;
  op_conf.set_name("inner_product_test");
  InnerProductOpConf* inner_product_conf = op_conf.mutable_innerproduct_conf();
  inner_product_conf->set_in("ip_in");
  inner_product_conf->set_out("ip_out");
  inner_product_conf->set_out_num(40);
  inner_product_conf->set_has_bias_term(has_bias_term);
  auto inner_product_op = OpMgr::Singleton()->ConstructOp(op_conf);

  OperatorProto op_proto;
  inner_product_op->ToProto(&op_proto);

  auto inner_product_kernel =
      new InnerProductKernel<device_type, FloatingPointType>();
  inner_product_kernel->InitFromOpProto(op_proto);

  return inner_product_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
void TestInnerProductKernel(bool has_bias_term) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  auto BnInOp2BlobPtr =
      BuildBnInOp2BlobPtr<device_type, FloatingPointType>(has_bias_term);

  auto inner_product_kernel =
      BuildInnerProductKernel<device_type, FloatingPointType>(has_bias_term);

  inner_product_kernel->Forward(ctx, BnInOp2BlobPtr);
  inner_product_kernel->Backward(ctx, BnInOp2BlobPtr);

  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
  KTCommon::CheckResult(BnInOp2BlobPtr, "weight_diff", "expected_weight_diff");
  if (has_bias_term) {
    KTCommon::CheckResult(BnInOp2BlobPtr, "bias_diff", "expected_bias_diff");
  }
}

}  // namespace test

TEST(InnerProductKernel, inner_product_kernel_cpu_with_bias) {
  test::TestInnerProductKernel<DeviceType::kCPU, float>(true);
  test::TestInnerProductKernel<DeviceType::kCPU, double>(true);
}

TEST(InnerProductKernel, inner_product_kernel_cpu_without_bias) {
  test::TestInnerProductKernel<DeviceType::kCPU, float>(false);
  test::TestInnerProductKernel<DeviceType::kCPU, double>(false);
}

TEST(InnerProductKernel, inner_product_kernel_gpu_with_bias) {
  test::TestInnerProductKernel<DeviceType::kGPU, float>(true);
  test::TestInnerProductKernel<DeviceType::kGPU, double>(true);
}

TEST(InnerProductKernel, inner_product_kernel_gpu_without_bias) {
  test::TestInnerProductKernel<DeviceType::kGPU, float>(false);
  test::TestInnerProductKernel<DeviceType::kGPU, double>(false);
}

}  // namespace oneflow
