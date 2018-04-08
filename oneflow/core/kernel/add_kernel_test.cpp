#include "oneflow/core/kernel/cwise_kernel_test.h"

namespace oneflow {

namespace test {

template<>
void SetCWiseOpConf<OperatorConf::OpTypeCase::kAddConf>(OperatorConf* op_conf,
                                                        int input_cnt) {
  auto* add_conf = op_conf->mutable_add_conf();
  FOR_RANGE(int, i, 0, input_cnt) {
    add_conf->add_in(std::string("in_") + std::to_string(i));
  }
}

DiffKernelImplTestCase* DiffAddKernelImpl(const std::string& job_type,
                                          const std::string& fw_or_bw,
                                          const std::string& cpp_type,
                                          const std::string& input_cnt_str) {
  return DiffCWiseKernelImpl<OperatorConf::OpTypeCase::kAddConf>(
      job_type, fw_or_bw, cpp_type, input_cnt_str);
}

TEST_DIFF_KERNEL_IMPL(DiffAddKernelImpl, (train)(predict), (forward)(backward),
                      OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, ARITHMETIC_DATA_TYPE_SEQ),
                      (1)(2)(5));

}  // namespace test

}  // namespace oneflow
