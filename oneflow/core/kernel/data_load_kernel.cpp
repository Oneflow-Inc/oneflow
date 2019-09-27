#include "oneflow/core/kernel/data_load_kernel.h"
#include "oneflow/core/data/dataset_manager.h"
#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

namespace {

void PreprocessBlob(const PreprocessConf& prep_conf, Blob* blob) {
#define MAKE_ENTRY(T, TVal)                                                 \
  {GetHashKey(TVal), [](const PreprocessConf& prep_conf, Blob* blob) {      \
     DoPreprocess<T>(prep_conf, blob->mut_dptr<T>(), blob->static_shape()); \
   }},
  static const HashMap<std::string, std::function<void(const PreprocessConf&, Blob*)>>
      preprocessers = {OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)};
#undef MAKE_ENTRY
  preprocessers.at(GetHashKey(blob->data_type()))(prep_conf, blob);
}

}  // namespace

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
  FOR_RANGE(int32_t, i, 0, op_attribute().output_bns_size()) {
    Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(i));
    const BlobConf& blob_conf = op_conf().data_load_conf().blobs(i);
    WriteDataToBlob(ctx.device_ctx, batch_data, blob_conf, out_blob);
  }
}

void DataLoadKernel::WriteDataToBlob(
    DeviceCtx* ctx, const std::vector<std::unique_ptr<data::DataInstance>>& data_inst_vec,
    const BlobConf& blob_conf, Blob* blob) const {
  using namespace data;
  bool is_contiguous = (blob->blob_desc().num_of_lod_levels() == 0);
  char* dptr = static_cast<char*>(blob->mut_dptr());
  Memset<DeviceType::kCPU>(ctx, dptr, 0, blob->ByteSizeOfDataContentField());
  Shape dense_shape;
  std::vector<std::vector<int64_t>> length_lod;
  if (is_contiguous) {
    const DataField* first = data_inst_vec.at(0)->GetField(blob_conf.data_source());
    first->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &dense_shape,
                      &length_lod);
    const int64_t elem_cnt = dense_shape.elem_cnt();
    if (!blob->blob_desc().is_dynamic()) {
      const int64_t exp_elem_cnt =
          std::accumulate(blob_conf.shape().dim().begin(), blob_conf.shape().dim().end(), 1,
                          std::multiplies<int64_t>());
      CHECK_EQ(elem_cnt, exp_elem_cnt);
    }
    MultiThreadLoop(data_inst_vec.size(), [&](int64_t n) {
      const DataField* data_field = data_inst_vec.at(n)->GetField(blob_conf.data_source());
      data_field->ToBuffer(dptr + n * elem_cnt, blob_conf.data_type());
      Shape shape;
      data_field->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &shape,
                             &length_lod);
      CHECK(dense_shape == shape);
    });
    dense_shape.Set(0, data_inst_vec.size());
  } else {
    for (const auto& data_inst_ptr : data_inst_vec) {
      const DataField* data_field = data_inst_ptr->GetField(blob_conf.data_source());
      size_t written_size = data_field->ToBuffer(dptr, blob_conf.data_type());
      data_field->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &dense_shape,
                             &length_lod);
      dptr += written_size;
    }
  }
  blob->dense_shape_mut_view().set_shape(dense_shape);
  blob->length_lod_mut_view().SetLength(length_lod);
  for (const auto& preprocess_conf : blob_conf.preprocess()) {
    PreprocessBlob(preprocess_conf, blob);
  }
}

REGISTER_KERNEL(OperatorConf::kDataLoadConf, DataLoadKernel);

}  // namespace oneflow
