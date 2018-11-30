#include <oneflow/core/rpc_service/codec.h>
#include "oneflow/core/kernel/decode_in_stream_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void DecodeInStreamKernel<device_type>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto& conn_buffer_pair =
      *static_cast<std::pair<int64_t, std::vector<rpc_service::blob_t>>*>(ctx.other);
  auto conn_id = std::to_string(std::get<0>(conn_buffer_pair));
  auto& buffers = std::get<1>(conn_buffer_pair);
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->set_dim0_valid_num(0, buffers.size());
  int64_t ele_cnt = out_blob->shape().Count(1);
  int64_t max_data_id_size = Global<JobDesc>::Get()->SizeOfOneDataId();

  for (int i = 0; i < buffers.size(); ++i) {
    OFRecord record;
    record.ParseFromArray(buffers[i].data(), buffers[i].size());
    if (out_blob->has_data_id_field()) {
      const Feature& feature = record.feature().at("data_id");
      CHECK_EQ(feature.bytes_list().value_size(), 1);
      const std::string& data_id_str = feature.bytes_list().value(0);
      CHECK_LE(data_id_str.size(), max_data_id_size);
      std::string data_id = data_id_str + "_" + conn_id;
      memcpy(out_blob->mut_data_id(i), data_id.c_str(), data_id.size());
      if (data_id_str.size() != max_data_id_size) {
        *(out_blob->mut_data_id(i) + data_id.size()) = '\0';
      }
    }

    auto it = record.feature().find("img_raw");
    if (it != record.feature().end() && out_blob->data_type() == DataType::kFloat) {
      auto in_ptr = it->second.float_list().value().data();
      float* out_ptr = out_blob->mut_dptr<float>() + i * ele_cnt;
      memcpy(out_ptr, in_ptr, ele_cnt);
    }
  }
  //        const DecodeRandomOpConf& conf = this->op_conf().decode_random_conf();
  //        if (is_init_ == false) {
  //            RandomFillBlob(ctx.device_ctx, device_type, conf.data_initializer(),
  //            this->GenNextRandomSeed(),
  //                           BnInOp2Blob("out"));
  //            is_init_ = true;
  //        }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDecodeInStreamConf, DecodeInStreamKernel);

}  // namespace oneflow