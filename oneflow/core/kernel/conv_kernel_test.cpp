#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
OpKernelTestCase* ConvTestCase(const std::string& job_type,
                               const std::string& forward_or_backward) {
  OpKernelTestCase* conv_test_case = new OpKernelTestCase();
  conv_test_case->mut_job_conf()->set_default_data_type(GetDataType<T>::value);
  conv_test_case->set_is_train(job_type == "train");
  conv_test_case->set_is_forward(forward_or_backward == "forward");
  conv_test_case->set_device_type(device_type);
  Conv3DOpConf* conv_conf =
      conv_test_case->mut_op_conf()->mutable_conv_3d_conf();
  conv_conf->set_padding("SAME");
  conv_conf->set_data_format("channels_first");
  conv_conf->add_kernel_size(3);
  conv_conf->add_kernel_size(3);
  conv_conf->add_kernel_size(3);
  conv_conf->add_strides(1);
  conv_conf->add_strides(1);
  conv_conf->add_strides(1);
  conv_conf->add_dilation_rate(1);
  conv_conf->add_dilation_rate(1);
  conv_conf->add_dilation_rate(1);
  conv_conf->set_use_bias(true);

  using KTC = KTCommon<device_type, T>;
  BlobDesc* in_blob_desc = new BlobDesc(Shape({1, 1, 4, 4, 4}),
                                        GetDataType<T>::value, false, false, 1);
  conv_test_case->InitBlob("in", KTC::CreateBlobWithSameVal(in_blob_desc, 1));

  BlobDesc* weight_blob_desc = new BlobDesc(
      Shape({1, 1, 3, 3, 3}), GetDataType<T>::value, false, false, 1);
  conv_test_case->InitBlob("weight",
                           KTC::CreateBlobWithSameVal(weight_blob_desc, 1));

  BlobDesc* bias_blob_desc =
      new BlobDesc(Shape({1, 1}), GetDataType<T>::value, false, false, 1);
  conv_test_case->InitBlob("bias",
                           KTC::CreateBlobWithSameVal(bias_blob_desc, 1));

  BlobDesc* bias_mul_blob_desc =
      new BlobDesc(Shape({1, 64}), GetDataType<T>::value, false, false, 1);
  conv_test_case->InitBlob("bias_multiplier",
                           KTC::CreateBlobWithSameVal(bias_mul_blob_desc, 1));

  conv_test_case->ForwardCheckBlob(
      "out", device_type,
      KTC::CreateBlobWithSpecifiedVal(
          in_blob_desc,
          {9.0,  13.0, 13.0, 9.0,  13.0, 19.0, 19.0, 13.0, 13.0, 19.0, 19.0,
           13.0, 9.0,  13.0, 13.0, 9.0,  13.0, 19.0, 19.0, 13.0, 19.0, 28.0,
           28.0, 19.0, 19.0, 28.0, 28.0, 19.0, 13.0, 19.0, 19.0, 13.0, 13.0,
           19.0, 19.0, 13.0, 19.0, 28.0, 28.0, 19.0, 19.0, 28.0, 28.0, 19.0,
           13.0, 19.0, 19.0, 13.0, 9.0,  13.0, 13.0, 9.0,  13.0, 19.0, 19.0,
           13.0, 13.0, 19.0, 19.0, 13.0, 9.0,  13.0, 13.0, 9.0}));

  conv_test_case->InitBlob("out_diff",
                           KTC::CreateBlobWithSameVal(in_blob_desc, 1));

  conv_test_case->BackwardCheckBlob(
      "in_diff", device_type,
      KTC::CreateBlobWithSpecifiedVal(
          in_blob_desc,
          {8,  12, 12, 8,  12, 18, 18, 12, 12, 18, 18, 12, 8,  12, 12, 8,
           12, 18, 18, 12, 18, 27, 27, 18, 18, 27, 27, 18, 12, 18, 18, 12,
           12, 18, 18, 12, 18, 27, 27, 18, 18, 27, 27, 18, 12, 18, 18, 12,
           8,  12, 12, 8,  12, 18, 18, 12, 12, 18, 18, 12, 8,  12, 12, 8}));

  conv_test_case->BackwardCheckBlob(
      "weight_diff", device_type,
      KTC::CreateBlobWithSpecifiedVal(
          weight_blob_desc,
          {27, 36, 27, 36, 48, 36, 27, 36, 27, 36, 48, 36, 48, 64,
           48, 36, 48, 36, 27, 36, 27, 36, 48, 36, 27, 36, 27}));

  conv_test_case->BackwardCheckBlob(
      "bias_diff", device_type, KTC::CreateBlobWithSameVal(bias_blob_desc, 64));

  return conv_test_case;
}

TEST_CPU_ONLY_OPKERNEL(ConvTestCase, FLOATING_DATA_TYPE_SEQ, (train),
                       (backward));

}  // namespace test

}  // namespace oneflow
