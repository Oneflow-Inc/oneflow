#include "oneflow/core/kernel/relu_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
OpKernelTestCase* ReluTestCase(bool is_train, bool is_forward) {
  OpKernelTestCase* relu_test_case = new OpKernelTestCase();
  relu_test_case->set_is_train(is_train);
  relu_test_case->set_is_forward(is_forward);
  relu_test_case->set_device_type(device_type);
  relu_test_case->mut_op_conf()->mutable_relu_conf();

  using KTC = KTCommon<device_type, T>;
  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 8}), GetDataType<T>::val, false, false, 1);
  relu_test_case->InitBlob(
      "in", KTC::CreateBlobWithSpecifiedVal(blob_desc,
                                            {1, -1, -2, 2, 0, 5, -10, 100}));
  relu_test_case->InitBlob(
      GenDiffBn("out"),
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {-8, 7, -6, 5, -4, 3, -2, 1}));
  relu_test_case->ForwardCheckBlob(
      "out", device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {1, 0, 0, 2, 0, 5, 0, 100}));
  relu_test_case->BackwardCheckBlob(
      GenDiffBn("in"), device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {-8, 0, 0, 5, 0, 3, 0, 1}));

  return relu_test_case;
}

#define MAKE_ENTRY(device_type, data_type_pair, is_train, is_forward)        \
  TEST(ReluKernel, OF_PP_JOIN(relu_, __COUNTER__, _, device_type, _,         \
                              OF_PP_PAIR_FIRST(data_type_pair), _, is_train, \
                              _, is_forward)) {                              \
    ReluTestCase<DeviceType::k##device_type,                                 \
                 OF_PP_PAIR_FIRST(data_type_pair)>(                          \
        std::string(#is_train) == "train",                                   \
        std::string(#is_forward) == "forward")                               \
        ->Run();                                                             \
  }

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    MAKE_ENTRY, (CPU), FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ,
    (train)(predict), (forward)(backward))

}  // namespace test

}  // namespace oneflow
