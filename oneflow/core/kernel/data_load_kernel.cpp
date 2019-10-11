#include "oneflow/core/kernel/data_load_kernel.h"
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
  data_loader_.reset(
      new data::DataLoader(this->op_conf().data_load_conf(), this->kernel_conf().data_load_conf()));
}

void DataLoadKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::shared_ptr<data::BatchDataInstance> batch_data = data_loader_->FetchBatch();
  FOR_RANGE(int32_t, i, 0, op_attribute().output_bns_size()) {
    Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(i));
    const BlobConf& blob_conf = op_conf().data_load_conf().blobs(i);
    WriteDataToBlob(ctx.device_ctx, batch_data, blob_conf, out_blob);
  }
}

void DataLoadKernel::WriteDataToBlob(DeviceCtx* ctx,
                                     std::shared_ptr<data::BatchDataInstance> batch_data,
                                     const BlobConf& blob_conf, Blob* blob) const {
  using namespace data;
  bool is_contiguous = (blob->blob_desc().num_of_lod_levels() == 0);
  char* dptr = static_cast<char*>(blob->mut_dptr());
  Memset<DeviceType::kCPU>(ctx, dptr, 0, blob->ByteSizeOfDataContentField());
  Shape dense_shape;
  if (is_contiguous) {
    const DataField* first = batch_data->Get(0)->GetField(blob_conf.data_source());
    first->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &dense_shape, nullptr);
    const int64_t elem_cnt = dense_shape.elem_cnt();
    if (!blob->blob_desc().is_dynamic()) {
      const int64_t exp_elem_cnt =
          std::accumulate(blob_conf.shape().dim().begin(), blob_conf.shape().dim().end(), 1,
                          std::multiplies<int64_t>());
      CHECK_EQ(elem_cnt, exp_elem_cnt);
    }
    MultiThreadLoop(batch_data->Size(), [&](int64_t n) {
      const DataField* data_field = batch_data->Get(n)->GetField(blob_conf.data_source());
      data_field->ToBuffer(dptr + n * elem_cnt, blob_conf.data_type());
      Shape shape;
      data_field->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &shape, nullptr);
      CHECK(dense_shape == shape);
    });
    dense_shape.Set(0, batch_data->Size());
  } else {
    std::vector<std::vector<int64_t>> length_lod;
    batch_data->ForEach([&](DataInstance* data_inst) {
      const DataField* data_field = data_inst->GetField(blob_conf.data_source());
      size_t written_size = data_field->ToBuffer(dptr, blob_conf.data_type());
      data_field->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &dense_shape,
                             &length_lod);
      dptr += written_size;
    });
    blob->length_lod_mut_view().SetLength(length_lod);
  }
  blob->dense_shape_mut_view().set_shape(dense_shape);
  // TODO: implement all preprocessor with transform
  for (const auto& preprocess_conf : blob_conf.preprocess()) {
    PreprocessBlob(preprocess_conf, blob);
  }
}

REGISTER_KERNEL(OperatorConf::kDataLoadConf, DataLoadKernel);

}  // namespace oneflow
