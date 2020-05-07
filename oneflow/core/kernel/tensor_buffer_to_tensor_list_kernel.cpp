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
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
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
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<typename T>
void TensorBufferToTensorListKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Shape shape(op_conf().tensor_buffer_to_tensor_list_conf().shape());
  int64_t batch_axis = op_conf().tensor_buffer_to_tensor_list_conf().batch_axis();

  TensorBackInserter back_inserter(out_blob);
  FOR_RANGE(int, i, 0, in_blob->shape().elem_cnt()) {
    const TensorBuffer& in_buffer = in_blob->dptr<TensorBuffer>()[i];
    DimVector dim_vec(1, 1);
    FOR_RANGE(int, j, 0, shape.NumAxes()) {
      if (j == batch_axis) {
        dim_vec.push_back(1);
      } else {
        dim_vec.push_back(shape.At(j));
      }
    }
    Shape out_shape(dim_vec);
    CHECK_EQ(shape.elem_cnt() % out_shape.elem_cnt(), 0);
    int64_t varying_dim = shape.elem_cnt() / out_shape.elem_cnt();
    out_shape.Set(batch_axis + 1, varying_dim);
    FullyMutTensorView* tensor_view = back_inserter.add_tensor();
    tensor_view->set_shape(out_shape);
    CHECK_EQ(tensor_view->shape().elem_cnt(), in_buffer.shape().elem_cnt());
    SwitchUtil<T>::SwitchCopyTensorBufferTo(SwitchCase(in_buffer.data_type()), in_buffer,
                                            tensor_view->mut_dptr<T>());
  }
}

#define REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(dtype)                      \
  NEW_REGISTER_KERNEL(OperatorConf::kTensorBufferToTensorListConf,               \
                      TensorBufferToTensorListKernel<dtype>)                     \
      .SetIsMatchedPred([](const KernelConf& conf) {                             \
        return (conf.op_attribute().op_conf().device_type() == DeviceType::kCPU) \
               && (conf.data_type() == GetDataType<dtype>::value);               \
      });

REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(int8_t)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(int32_t)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(int64_t)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(float)
REGISTER_TENSOR_BUFFER_TO_TENSOR_LIST_KERNEL(double)

}  // namespace oneflow
