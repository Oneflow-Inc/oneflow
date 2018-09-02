#include "oneflow/core/kernel/pb_boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
void PbBoxingKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (boxing_conf().in_box_case() == BoxingOpConf::kConcatBox
      && boxing_conf().out_box_case() == BoxingOpConf::kSplitBox) {
    CHECK_EQ(boxing_conf().concat_box().axis(), boxing_conf().split_box().axis());
    RecordContentIterator<T> in_ter(BnInOp2Blob, &this->op_attribute().pb_input_bns(),
                                    boxing_conf().concat_box().axis());
    RecordContentIterator<T> out_ter(BnInOp2Blob, &this->op_attribute().pb_output_bns(),
                                     boxing_conf().split_box().axis());
    while (true) {
      T* in_record = in_ter.GetNext();
      T* out_record = out_ter.GetNext();
      if (in_record == nullptr && out_record == nullptr) { break; }
      if (in_record != nullptr && out_record != nullptr) {
        *out_record = *in_record;
      } else {
        UNIMPLEMENTED();
      }
    }
  } else {
    UNIMPLEMENTED();
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kPbBoxingConf, PbBoxingKernel, PB_LIST_DATA_TYPE_SEQ);

}  // namespace oneflow
