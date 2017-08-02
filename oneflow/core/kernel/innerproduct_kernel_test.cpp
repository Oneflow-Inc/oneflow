#include "oneflow/core/kernel/innerproduct_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
    bool has_bias_term) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  FloatingPointType in_mat[] = {1, 2, 3, 4, 5, 6, 7, 8};
  FloatingPointType weight_mat[] = {5, 4, 5, 3, 2, 1, 7, 0, 1, 1, 9, 8};
  FloatingPointType bias_mat[] = {2, 3, 5};
  FloatingPointType bias_multiplier_mat[] = {1, 1};

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

  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({2, 1, 2, 2}, in_mat);
  (*bn2blob_ptr)["weight"] = KTCommon::CreateBlobWithVector({3, 4}, weight_mat);
  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithRandomValue({2, 3});
  (*bn2blob_ptr)["out_diff"] = (*bn2blob_ptr)["out"];
  (*bn2blob_ptr)["in_diff"] = KTCommon::CreateBlobWithRandomValue({2, 1, 2, 2});
  (*bn2blob_ptr)["weight_diff"] = KTCommon::CreateBlobWithRandomValue({3, 4});

  if (has_bias_term) {
    (*bn2blob_ptr)["bias"] = KTCommon::CreateBlobWithVector({1, 3}, bias_mat);
    (*bn2blob_ptr)["bias_multiplier"] =
        KTCommon::CreateBlobWithVector({2, 1}, bias_multiplier_mat);
    (*bn2blob_ptr)["bias_diff"] = KTCommon::CreateBlobWithRandomValue({1, 3});
    (*bn2blob_ptr)["expected_bias_diff"] =
        KTCommon::CreateBlobWithVector({1, 3}, expected_bias_diff_mat);
    (*bn2blob_ptr)["expected_out"] =
        KTCommon::CreateBlobWithVector({2, 3}, expected_out_mat);
    (*bn2blob_ptr)["expected_in_diff"] =
        KTCommon::CreateBlobWithVector({2, 1, 2, 2}, expected_in_diff_mat);
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
std::function<Blob*(const std::string&)> BuildEmptyMdlForFill() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  (*bn2blob_ptr)["weight"] = KTCommon::CreateBlobWithRandomValue({100, 100});
  (*bn2blob_ptr)["bias"] = KTCommon::CreateBlobWithRandomValue({100, 100});
  (*bn2blob_ptr)["bias_multiplier"] =
      KTCommon::CreateBlobWithRandomValue({100, 100});

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildInnerProductKernel(bool has_bias_term, FillConf* fill_conf) {
  OperatorConf op_conf;
  op_conf.set_name("inner_product_test");
  InnerProductOpConf* inner_product_conf = op_conf.mutable_innerproduct_conf();
  inner_product_conf->set_in("ip_in");
  inner_product_conf->set_out("ip_out");
  inner_product_conf->set_out_num(40);
  inner_product_conf->set_has_bias_term(has_bias_term);

  if (fill_conf != nullptr) {
    inner_product_conf->mutable_weight_fill()->CopyFrom(*fill_conf);
    inner_product_conf->mutable_bias_fill()->CopyFrom(*fill_conf);
  }

  auto inner_product_op = ConstructOp(op_conf);

  OperatorProto op_proto;
  inner_product_op->ToProto(&op_proto);

  auto inner_product_kernel =
      new InnerProductKernel<device_type, FloatingPointType>();
  inner_product_kernel->InitFromOpProto(op_proto);

  return inner_product_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
void IpKernelFwAndBp(bool has_bias_term) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  auto BnInOp2Blob =
      BuildBnInOp2BlobPtr<device_type, FloatingPointType>(has_bias_term);

  auto inner_product_kernel =
      BuildInnerProductKernel<device_type, FloatingPointType>(has_bias_term,
                                                              nullptr);

  inner_product_kernel->Forward(ctx, BnInOp2Blob);
  inner_product_kernel->Backward(ctx, BnInOp2Blob);

  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2Blob, "out", "expected_out");
  KTCommon::CheckResult(BnInOp2Blob, "in_diff", "expected_in_diff");
  KTCommon::CheckResult(BnInOp2Blob, "weight_diff", "expected_weight_diff");
  if (has_bias_term) {
    KTCommon::CheckResult(BnInOp2Blob, "bias_diff", "expected_bias_diff");
  }
}

template<DeviceType device_type, typename FloatingPointType>
void IpKernelFillMdlAndMdlTmp(FillConf* fill_conf) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  auto BnInOp2Blob = BuildEmptyMdlForFill<device_type, FloatingPointType>();

  auto inner_product_kernel =
      BuildInnerProductKernel<device_type, FloatingPointType>(true, fill_conf);
  inner_product_kernel->InitModelAndModelTmpBlobs(
      ctx, ParallelPolicy::kDataParallel, 0, 0, nullptr, BnInOp2Blob);

  KTCommon::SyncStream(&ctx);

  KTCommon::CheckFillResult(*BnInOp2Blob("weight"), *fill_conf);
  KTCommon::CheckFillResult(*BnInOp2Blob("bias"), *fill_conf);
  KTCommon::CheckFillResult(*BnInOp2Blob("bias_multiplier"), *fill_conf);
}

}  // namespace

}  // namespace test

TEST(InnerProductKernel, inner_product_kernel_cpu_with_bias) {
  test::IpKernelFwAndBp<DeviceType::kCPU, float>(true);
  test::IpKernelFwAndBp<DeviceType::kCPU, double>(true);
}

TEST(InnerProductKernel, inner_product_kernel_cpu_without_bias) {
  test::IpKernelFwAndBp<DeviceType::kCPU, float>(false);
  test::IpKernelFwAndBp<DeviceType::kCPU, double>(false);
}

TEST(InnerProductKernel, inner_product_kernel_gpu_with_bias) {
  test::IpKernelFwAndBp<DeviceType::kGPU, float>(true);
  test::IpKernelFwAndBp<DeviceType::kGPU, double>(true);
}

TEST(InnerProductKernel, inner_product_kernel_gpu_without_bias) {
  test::IpKernelFwAndBp<DeviceType::kGPU, float>(false);
  test::IpKernelFwAndBp<DeviceType::kGPU, double>(false);
}

TEST(InnerProductKernel, fill_model_in_cpu_with_constant) {
  FillConf fill_conf;
  fill_conf.mutable_constant_conf()->set_value(1.0f);
  test::IpKernelFillMdlAndMdlTmp<DeviceType::kCPU, float>(&fill_conf);
  test::IpKernelFillMdlAndMdlTmp<DeviceType::kCPU, double>(&fill_conf);
}

TEST(InnerProductKernel, fill_model_in_gpu_with_constant) {
  FillConf fill_conf;
  fill_conf.mutable_constant_conf()->set_value(1.0f);
  test::IpKernelFillMdlAndMdlTmp<DeviceType::kGPU, float>(&fill_conf);
  test::IpKernelFillMdlAndMdlTmp<DeviceType::kGPU, double>(&fill_conf);
}

}  // namespace oneflow
