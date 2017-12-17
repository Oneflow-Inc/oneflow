#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/operator/clone_op.h"
#include "oneflow/core/kernel/clone_kernel.h"

namespace oneflow {

namespace test {

std::shared_ptr<Operator> CreateCloneOp(int out_num) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  op_conf.mutable_clone_conf()->set_out_num(out_num);
  op_conf.mutable_clone_conf()->set_lbn("clone_lbn");
  return ConstructOp(op_conf);
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string)> BuildBnInOp2BlobFunc(int out_num) {
  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 3, 2}), GetDataType<T>::val, false);
  HashMap<std::string, BlobInitConf> bn2blob_init_conf = {
      {"in", BlobInitConf(1.f, blob_desc)},
      {GenDiffBn("in"), BlobInitConf(blob_desc)},
      {"in_diff_expected", BlobInitConf(out_num * 1.f, blob_desc)}};
  FOR_RANGE(int, i, 0, out_num) {
    bn2blob_init_conf.insert(
        {"out_" + std::to_string(i), BlobInitConf(blob_desc)});
    bn2blob_init_conf.insert(
        {"out_" + std::to_string(i) + "_diff", BlobInitConf(1.f, blob_desc)});
  }
  return KTCommon<device_type, T>::ConstructBnInOp2BlobFunc(bn2blob_init_conf);
}

template<DeviceType device_type, typename T>
void DoCloneKernelTest(int out_num) {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto clone_op = CreateCloneOp(out_num);
  auto bn2blobdesc_func = ConstructBn2BlobDescFunc(clone_op);
  KernelConf kernel_conf;
  clone_op->GenKernelConf(bn2blobdesc_func, true, nullptr, &kernel_conf);
  auto clone_kernel = new CloneKernel<device_type, T>();
  clone_kernel->Init(nullptr, kernel_conf);

  auto BnInOp2BlobFunc = BuildBnInOp2BlobFunc<device_type, T>(out_num);
  clone_kernel->Forward(ctx, BnInOp2BlobFunc);
  clone_kernel->Backward(ctx, BnInOp2BlobFunc);
  SyncStream<device_type>(&ctx);

  FOR_RANGE(size_t, i, 0, out_num) {
    KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, "in",
                                          "out_" + std::to_string(i));
  }

  KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, GenDiffBn("in"),
                                        "in_diff_expected");
}

template<typename T, bool has_data_id>
void DoCloneOpTest(int out_num) {
  auto clone_op = CreateCloneOp(out_num);
  auto bn2blobdesc_func = ConstructBn2BlobDescFunc(clone_op);
  BlobDesc* in_blob_desc = bn2blobdesc_func("in");
  in_blob_desc->mut_shape().dim_vec_ = {3, 4};
  in_blob_desc->set_data_type(GetDataType<T>::val);
  in_blob_desc->set_has_data_id(has_data_id);

  clone_op->InferBlobDescs(bn2blobdesc_func, nullptr);

  for (const std::string& obn : clone_op->output_bns()) {
    const BlobDesc* out_blob_desc = bn2blobdesc_func(obn);
    ASSERT_TRUE(*in_blob_desc == *out_blob_desc);
  }
}

template<DeviceType device_type, typename T, bool has_data_id>
void DoCloneOpKernelTest(int out_num) {
  DoCloneOpTest<T, has_data_id>(out_num);
  DoCloneKernelTest<device_type, T>(out_num);
}

template<DeviceType device_type, typename T, bool has_data_id>
void TestCloneOpKernel() {
  JobConf job_conf;
  job_conf.set_default_data_type(GetDataType<T>::val);
  JobDesc::NewSingleton();
  JobDesc::Singleton()->job_conf_ = job_conf;

  int out_num = 3;
  DoCloneOpKernelTest<device_type, T, has_data_id>(out_num);
}

TEST(CloneOpKernel, op_and_kernel_test) {
#define MAKE_ENTRY(device_type, data_type_pair, has_data_id)       \
  TestCloneOpKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair), \
                    has_data_id>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace test
}  // namespace oneflow
