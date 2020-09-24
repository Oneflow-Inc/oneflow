/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

namespace {

template<typename S, typename D>
void CopyTensorBufferTo(const TensorBuffer& src, D* dst) {
  CopyElem(src.data<S>(), dst, src.elem_cnt());
}

template<typename D>
struct SwitchUtil final {
#define MAKE_COPY_ELEM_SWITCH_ENTRY(func_name, S) func_name<S, D>
  DEFINE_STATIC_SWITCH_FUNC(void, CopyTensorBufferTo, MAKE_COPY_ELEM_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ));
#undef MAKE_COPY_ELEM_SWITCH_ENTRY
};

}  // namespace

template<typename T>
class TensorBufferToTensorListKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorBufferToTensorListKernel);
  TensorBufferToTensorListKernel() = default;
  ~TensorBufferToTensorListKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)>) const override;

  void ForwardHeader(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    CHECK(!this->kernel_conf().need_do_opaque_header());
    if (this->kernel_conf().need_do_shape()) { ForwardShape(ctx, BnInOp2Blob); }
  }
};

template<typename T>
void TensorBufferToTensorListKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Shape conf_shape(op_conf().tensor_buffer_to_tensor_list_conf().shape());

  TensorBackInserter back_inserter(out_blob);
  back_inserter.ReserveOneEmptyTensorList();
  FOR_RANGE(int, i, 0, in_blob->shape().elem_cnt()) {
    const TensorBuffer& in_buffer = in_blob->dptr<TensorBuffer>()[i];
    CHECK_EQ(in_buffer.shape().NumAxes(), conf_shape.NumAxes())
        << "in_buffer.shape " << in_buffer.shape().ToString() << ", conf_shape "
        << conf_shape.ToString();
    CHECK_LE(in_buffer.shape().elem_cnt(), conf_shape.elem_cnt())
        << "in_buffer.shape " << in_buffer.shape().ToString() << ", conf_shape "
        << conf_shape.ToString();
    DimVector dim_vec = in_buffer.shape().dim_vec();
    dim_vec.insert(dim_vec.begin(), 1);
    FullyMutTensorView* tensor_view = back_inserter.add_tensor();
    tensor_view->set_shape(Shape(dim_vec));
    SwitchUtil<T>::SwitchCopyTensorBufferTo(SwitchCase(in_buffer.data_type()), in_buffer,
                                            tensor_view->mut_dptr<T>());
  }
  CHECK_EQ(out_blob->total_num_of_tensors(), in_blob->shape().elem_cnt());
}

#define REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(dtype)          \
  NEW_REGISTER_KERNEL(OperatorConf::kTensorBufferToTensorListConf,   \
                      TensorBufferToTensorListKernel<dtype>)         \
      .SetIsMatchedPred([](const KernelConf& conf) {                 \
        return (conf.op_attribute().op_conf().device_tag() == "cpu") \
               && (conf.data_type() == GetDataType<dtype>::value);   \
      });

REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(char)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(int8_t)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(uint8_t)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(int32_t)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(int64_t)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(float)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(double)

}  // namespace oneflow
