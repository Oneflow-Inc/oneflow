#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class TensorListSplitKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorListSplitKernel);
  TensorListSplitKernel() = default;
  ~TensorListSplitKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    CHECK(in_blob->blob_desc().is_tensor_list());
    CHECK_EQ(in_blob->total_num_of_tensors(), this->op_attribute().output_bns_size());
    TensorView in_tensor = in_blob->BeginTensor();
    for (const auto& obn : this->op_attribute().output_bns()) {
      Blob* out_blob = BnInOp2Blob(obn);
      TensorBackInserter tensor_back_inserter(out_blob);
      tensor_back_inserter.ReserveOneEmptyTensorList();
      FullyMutTensorView* out_tensor_view = tensor_back_inserter.add_tensor();
      DimVector dim_vec;
      in_tensor.shape().ToDimVector(&dim_vec);
      dim_vec.erase(dim_vec.begin());
      out_tensor_view->set_shape(Shape(dim_vec));
      Memcpy<device_type>(ctx.device_ctx, out_tensor_view->mut_dptr(), in_tensor.dptr(),
                          in_tensor.ByteSize());
      in_blob->MoveToNextTensor(&in_tensor);
    }
    CHECK(in_blob->IsEndTensor(in_tensor));
  }

  void ForwardHeader(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    // do nothing
  }
};

#define REGISTER_TENSOR_LIST_SPLIT_KERNEL(device, dtype)                                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kTensorListSplitConf, device, dtype, \
                                        TensorListSplitKernel<device, dtype>)

#define REGISTER_TENSOR_LIST_SPLIT_KERNEL_WITH_DEVICE_AND_DTYPE_PAIR(device, dtype_pair) \
  REGISTER_TENSOR_LIST_SPLIT_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_TENSOR_LIST_SPLIT_KERNEL_WITH_DEVICE_AND_DTYPE_PAIR,
                                 DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
