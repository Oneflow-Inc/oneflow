#include "oneflow/core/kernel/innerproduct_kernel.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
    bool has_bias_term) {
  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  BlobDesc* blob_desc2122 =
      new BlobDesc(Shape({2, 1, 2, 2}), GetDataType<T>::val, false);
  BlobDesc* blob_desc34 =
      new BlobDesc(Shape({3, 4}), GetDataType<T>::val, false);
  BlobDesc* blob_desc23 =
      new BlobDesc(Shape({2, 3}), GetDataType<T>::val, false);
  BlobDesc* blob_desc13 =
      new BlobDesc(Shape({1, 3}), GetDataType<T>::val, false);
  BlobDesc* blob_desc21 =
      new BlobDesc(Shape({2, 1}), GetDataType<T>::val, false);
  (*bn2blob_ptr)["in"] = KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(
      blob_desc2122, {1, 2, 3, 4, 5, 6, 7, 8});
  (*bn2blob_ptr)["weight"] =
      KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(
          blob_desc34, {5, 4, 5, 3, 2, 1, 7, 0, 1, 1, 9, 8});
  (*bn2blob_ptr)["out"] =
      KTCommon<device_type, T>::CreateBlobWithRandomVal(blob_desc23);
  (*bn2blob_ptr)[GenDiffBn("out")] = (*bn2blob_ptr)["out"];
  (*bn2blob_ptr)[GenDiffBn("in")] =
      KTCommon<device_type, T>::CreateBlobWithRandomVal(blob_desc2122);
  (*bn2blob_ptr)[GenDiffBn("weight")] =
      KTCommon<device_type, T>::CreateBlobWithRandomVal(blob_desc34);

  if (has_bias_term) {
    (*bn2blob_ptr)["bias"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(blob_desc13,
                                                             {2, 3, 5});
    (*bn2blob_ptr)["bias_multiplier"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(blob_desc21,
                                                             {1, 1});
    (*bn2blob_ptr)[GenDiffBn("bias")] =
        KTCommon<device_type, T>::CreateBlobWithRandomVal(blob_desc13);
    (*bn2blob_ptr)["expected_bias_diff"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(blob_desc13,
                                                             {152, 96, 210});
    (*bn2blob_ptr)["expected_out"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(
            blob_desc23, {42, 28, 67, 110, 68, 143});
    (*bn2blob_ptr)["expected_in_diff"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(
            blob_desc2122, {333, 263, 1009, 662, 829, 651, 2313, 1474});
    (*bn2blob_ptr)["expected_weight_diff"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(
            blob_desc34,
            {592, 744, 896, 1048, 368, 464, 560, 656, 782, 992, 1202, 1412});
  } else {
    (*bn2blob_ptr)["expected_out"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(
            blob_desc23, {40, 25, 62, 108, 65, 138});
    (*bn2blob_ptr)["expected_in_diff"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(
            blob_desc2122, {312, 247, 933, 616, 808, 635, 2237, 1428});
    (*bn2blob_ptr)["expected_weight_diff"] =
        KTCommon<device_type, T>::CreateBlobWithSpecifiedVal(
            blob_desc34,
            {580, 728, 876, 1024, 350, 440, 530, 620, 752, 952, 1152, 1352});
  }
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildEmptyMdlForFill() {
  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  BlobDesc* blob_desc_fill =
      new BlobDesc(Shape({100, 100}), GetDataType<T>::val, false);
  (*bn2blob_ptr)["weight"] =
      KTCommon<device_type, T>::CreateBlobWithRandomVal(blob_desc_fill);
  (*bn2blob_ptr)["bias"] =
      KTCommon<device_type, T>::CreateBlobWithRandomVal(blob_desc_fill);
  (*bn2blob_ptr)["bias_multiplier"] =
      KTCommon<device_type, T>::CreateBlobWithRandomVal(blob_desc_fill);

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename T>
Kernel* BuildInnerProductKernel(bool has_bias_term, const FillConf* fill_conf) {
  OperatorConf op_conf;
  op_conf.set_name("inner_product_test");
  InnerProductOpConf* inner_product_conf = op_conf.mutable_innerproduct_conf();
  inner_product_conf->mutable_in()->set_name("ip_in");
  inner_product_conf->mutable_in()->set_data_type(GetDataType<T>::val);
  inner_product_conf->mutable_out()->set_name("ip_out");
  inner_product_conf->mutable_out()->set_data_type(GetDataType<T>::val);
  inner_product_conf->set_out_num(3);
  inner_product_conf->set_has_bias_term(has_bias_term);

  if (fill_conf != nullptr) {
    inner_product_conf->mutable_weight_fill()->CopyFrom(*fill_conf);
    inner_product_conf->mutable_bias_fill()->CopyFrom(*fill_conf);
  }

  auto inner_product_op = ConstructOp(op_conf);

  OperatorProto op_proto;
  inner_product_op->ToProto(&op_proto);

  auto inner_product_kernel = new InnerProductKernel<device_type, T>();
  inner_product_kernel->InitFromOpProto(op_proto);

  return inner_product_kernel;
}

template<DeviceType device_type, typename T>
void IpKernelFwAndBp(bool has_bias_term) {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2Blob = BuildBnInOp2BlobPtr<device_type, T>(has_bias_term);

  auto inner_product_kernel =
      BuildInnerProductKernel<device_type, T>(has_bias_term, nullptr);

  inner_product_kernel->Forward(ctx, BnInOp2Blob);
  inner_product_kernel->Backward(ctx, BnInOp2Blob);

  SyncStream<device_type>(&ctx);

  KTCommon<device_type, T>::CheckResult(BnInOp2Blob, "out", "expected_out");
  KTCommon<device_type, T>::CheckResult(BnInOp2Blob, "in_diff",
                                        "expected_in_diff");
  KTCommon<device_type, T>::CheckResult(BnInOp2Blob, "weight_diff",
                                        "expected_weight_diff");
  if (has_bias_term) {
    KTCommon<device_type, T>::CheckResult(BnInOp2Blob, "bias_diff",
                                          "expected_bias_diff");
  }
}

template<DeviceType device_type, typename T>
void IpKernelFillMdlAndMdlTmp(const FillConf* fill_conf) {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2Blob = BuildEmptyMdlForFill<device_type, T>();

  auto inner_product_kernel =
      BuildInnerProductKernel<device_type, T>(true, fill_conf);
  inner_product_kernel->InitModelBlobs(ctx, ParallelPolicy::kDataParallel, 0, 0,
                                       nullptr, BnInOp2Blob);
  inner_product_kernel->InitModelTmpBlobs(ctx, BnInOp2Blob);

  SyncStream<device_type>(&ctx);

  KTCommon<device_type, T>::CheckFillResult(BnInOp2Blob("weight"), *fill_conf);
  KTCommon<device_type, T>::CheckFillResult(BnInOp2Blob("bias"), *fill_conf);
  KTCommon<device_type, T>::CheckFillResult(BnInOp2Blob("bias_multiplier"),
                                            *fill_conf);
}

}  // namespace

}  // namespace test

TEST(InnerProductKernel, IpKernelFwAndBp) {
#define MAKE_ENTRY(device_type, type_pair, has_bias_term)          \
  test::IpKernelFwAndBp<device_type, OF_PP_PAIR_FIRST(type_pair)>( \
      has_bias_term);
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ, BOOL_SEQ)
#undef MAKE_ENTRY
}

TEST(InnerProductKernel, FillModelConstant) {
  FillConf fill_conf;
  fill_conf.mutable_constant_conf()->set_value(1.0f);
#define MAKE_ENTRY(device_type, type_pair)                                  \
  test::IpKernelFillMdlAndMdlTmp<device_type, OF_PP_PAIR_FIRST(type_pair)>( \
      &fill_conf);
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ)
}

}  // namespace oneflow
