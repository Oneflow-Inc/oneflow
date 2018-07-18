
#ifndef ONEFLOW_KERNEL_COMMON_HPP
#define ONEFLOW_KERNEL_COMMON_HPP
namespace oneflow{
    //function object for add/clone kernel
    template<DeviceType device_type, typename T, typename... Args>
    inline std::enable_if_t<std::is_same<T, float>::value> AdditionAssign(float, DeviceCtx* device_ctx, Blob* out, Args... in) {
        KernelUtil<device_type, float>::AdditionAssign(
                device_ctx, out->shape().elem_cnt(), out->mut_dptr<float>(), in->template dptr<float>()...);
    }
    template<DeviceType device_type, typename T, typename... Args>
    inline std::enable_if_t<std::is_same<T, double>::value> AdditionAssign(double, DeviceCtx* device_ctx, Blob* out, Args... in) {
        KernelUtil<device_type, double>::AdditionAssign(
                device_ctx, out->shape().elem_cnt(), out->mut_dptr<double>(), in->template dptr<double>()...);
    }

    template<DeviceType device_type, typename T, typename... Args>
    inline void AdditionAssign(...) {
        UNIMPLEMENTED();
    }

    template<bool in, DeviceType device_type, typename T, typename U>
    struct KernelFunction {
        template<typename V>
        void operator()(V v) {
            AdditionAssignImpl(std::make_index_sequence<decltype(v)::value>());
        }

        template<size_t... Idx>
        void AdditionAssignImpl(std::index_sequence<Idx...>) {
            if (in) {
                AdditionAssign<device_type, T>(T(), device_ctx_, diff_blob_,
                                            BnInOp2Blob_(u_->op_attribute().input_bns(offset_ + Idx))...);
            } else {
                AdditionAssign<device_type, T>(
                        T(), device_ctx_, diff_blob_,
                        BnInOp2Blob_(u_->op_attribute().output_diff_bns(offset_ + Idx))...);
            }
        }

        void AdditionAssignImpl(std::index_sequence<>) {}

        Blob* diff_blob_;
        std::function<Blob*(const std::string&)> BnInOp2Blob_;
        DeviceCtx* device_ctx_;
        int32_t offset_;
        U u_;
    };
}
#endif //ONEFLOW_KERNEL_COMMON_HPP
