#include "oneflow/core/kernel/data_load_kernel.h"
#include "oneflow/core/data/dataset_manager.h"

namespace oneflow {

void DataLoadKernel::VirtualKernelInit() {
  using namespace data;
  const DataLoadOpConf& op_conf = this->op_conf().data_load_conf();
  std::shared_ptr<Dataset> dataset =
      Global<DatasetManager>::Get()->GetOrCreateDataset(op_conf.dataset());
  data_loader_.reset(
      new DataLoader(this->kernel_conf().data_load_conf(), dataset, op_conf.batch_cache_size()));
}

void DataLoadKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::vector<std::unique_ptr<data::DataInstance>> batch_data = data_loader_->FetchBatch();
}

void DataLoadKernel::WriteDataToBlob(
    DeviceCtx* ctx, const std::vector<std::unique_ptr<data::DataInstance>>& data_inst_vec,
    const BlobConf& blob_conf, Blob* out_blob) const {
  using namespace data;
  // const std::string& field_name = blob_conf.field();
  // size_t elem_size = GetSizeOfDataType(blob_conf.data_type());
  // size_t out_blob_inst_elem_cnt = ShapeCount(blob_conf.shape());
  // size_t data_field_elem_cnt = data_inst_vec.at(0).GetField(field_name)->ElemCnt();
  // bool dense = out_blob->has_instance_shape_field();
  // const DataInstanceProto& data_inst_proto =
  //     kernel_conf().decode_ofrecord_conf().data_instance_conf();

  char* dptr = static_cast<char*>(out_blob->mut_dptr());
  Memset<DeviceType::kCPU>(ctx, dptr, 0, out_blob->ByteSizeOfDataContentField());

  for (const auto& data_inst_ptr : data_inst_vec) {
    const DataField* data_field = data_inst_ptr->GetField(blob_conf.data_source());
    size_t written_size = data_field->ToBuffer(dptr, blob_conf.data_type());
    dptr += written_size;
    //
    // data_field->InferShape(const ShapeProto& shape, );
  }

  // MultiThreadLoop(data_inst_vec.size(), [&](int64_t n) {
  //   const data::DataField* data_field = data_inst_vec.at(n)->GetField(field_name);
  //   size_t one_inst_size = (dense ? data_field_elem_cnt : out_blob_inst_elem_cnt) * elem_size;
  //   char* dptr = static_cast<char*>(out_blob->mut_dptr()) + one_inst_size * n;
  //   size_t written_size = data_field->ToBuffer(dptr, data_inst_proto.fields().at(field_name));
  //   if (dense) {
  //     CHECK_EQ(one_inst_size, written_size);
  //   } else {
  //     CHECK_GE(one_inst_size, written_size);
  //     Memset<DeviceType::kCPU>(ctx, dptr + written_size, 0, one_inst_size - written_size);
  //     if (blob_conf.encode_case().has_raw()) {
  //       if (out_blob->has_dim1_valid_num_field()) {
  //         size_t one_col_size = ShapeCount(blob_conf.shape(), 1) * elem_size;
  //         out_blob->set_dim1_valid_num(n, written_size / one_col_size);
  //       }
  //     } else if (blob_conf.encode_case().has_bytes_list()) {
  //       using DataFieldT = typename data::DataFieldTrait<DataCodec::kBytesList, DataType::kChar,
  //                                                        DataCase::kSegmentation>::type;
  //       const auto* spec_field = dynamic_cast<const DataFieldT*>(data_field);
  //       CHECK_NOTNULL(spec_field);
  //       if (out_blob->has_dim1_valid_num_field()) {
  //         size_t one_col_size = spec_field->pb_max_size();
  //         out_blob->set_dim1_valid_num(n, written_size / one_col_size);
  //       }
  //       if (out_blob->has_dim2_valid_num_field()) {
  //         FOR_RANGE(size_t, i, 0, spec_field->pb_sizes().size()) {
  //           out_blob->set_dim2_valid_num(n, i, spec_field->pb_sizes().at(i));
  //         }
  //       }
  //     }
  //   }
  // });
  // if (dense) {
  //   size_t total_written_size = data_field_elem_cnt * data_inst_vec.size() * elem_size;
  //   Memset<DeviceType::kCPU>(ctx, static_cast<char*>(out_blob->mut_dptr()) + total_written_size,
  //   0,
  //                            out_blob->ByteSizeOfDataContentField() - total_written_size);
  // }
  // FOR_RANGE(size_t, j, 0, blob_conf.preprocess_size()) {
  //   PreprocessBlob(blob_conf.preprocess(j), out_blob);
  // }
}

REGISTER_KERNEL(OperatorConf::kDataLoadConf, DataLoadKernel);

}  // namespace oneflow
