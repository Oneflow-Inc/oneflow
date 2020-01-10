#include "oneflow/core/kernel/data_load_kernel.h"
#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/register/lod_view.h"
#include "oneflow/core/nvtx3/nvToolsExt.h"

namespace oneflow {

void DataLoadKernel::VirtualKernelInit() {
  data_loader_.reset(
      new data::DataLoader(this->op_conf().data_load_conf(), this->kernel_conf().data_load_conf()));
}

void DataLoadKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const std::string mark("DataLoadKernel::Forward FetchBatch");
  nvtxRangePush(mark.c_str());
  auto batch_data = data_loader_->FetchBatch();
  nvtxRangePop();
  FOR_RANGE(int32_t, i, 0, op_attribute().output_bns_size()) {
    Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(i));
    const BlobConf& blob_conf = op_conf().data_load_conf().blobs(i);
    WriteDataToBlob(ctx.device_ctx, batch_data, blob_conf, out_blob);
  }
}

void DataLoadKernel::WriteDataToBlob(DeviceCtx* ctx,
                                     std::shared_ptr<std::vector<data::DataInstance>> batch_data,
                                     const BlobConf& blob_conf, Blob* blob) const {
  using namespace data;
  bool is_contiguous = (blob->blob_desc().num_of_lod_levels() == 0);
  char* dptr = static_cast<char*>(blob->mut_dptr());
  Memset<DeviceType::kCPU>(ctx, dptr, 0, blob->AlignedByteSizeOfBlobBody());
  Shape dense_shape;
  if (is_contiguous) {
    const DataField* first = batch_data->at(0).GetField(blob_conf.data_source());
    first->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &dense_shape, nullptr);
    const int64_t elem_cnt = dense_shape.elem_cnt();
    if (!blob->blob_desc().is_dynamic()) {
      const int64_t exp_elem_cnt =
          std::accumulate(blob_conf.shape().dim().begin(), blob_conf.shape().dim().end(), 1,
                          std::multiplies<int64_t>());
      CHECK_EQ(elem_cnt, exp_elem_cnt);
    }
    FOR_RANGE(size_t, n, 0, batch_data->size()) {
      const DataField* data_field = batch_data->at(n).GetField(blob_conf.data_source());
      size_t elem_bytes_size = GetSizeOfDataType(blob_conf.data_type());
      data_field->ToBuffer(dptr + n * elem_cnt * elem_bytes_size, blob_conf.data_type());
      Shape shape;
      data_field->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &shape, nullptr);
      CHECK(dense_shape == shape);
    }
    DimVector dense_shape_vec = dense_shape.dim_vec();
    dense_shape_vec.insert(dense_shape_vec.begin(), batch_data->size());
    dense_shape = Shape(dense_shape_vec);
  } else {
    LoDTree lod_tree;
    for (DataInstance& data_inst : *batch_data) {
      const DataField* data_field = data_inst.GetField(blob_conf.data_source());
      size_t written_size = data_field->ToBuffer(dptr, blob_conf.data_type());
      dptr += written_size;

      Shape inst_shape;
      LoDTree* inst_lod_tree = lod_tree.mutable_children()->Add();
      data_field->InferShape(blob_conf.shape(), blob_conf.variable_length_axes(), &inst_shape,
                             inst_lod_tree);
      if (dense_shape.elem_cnt() == 0) {
        dense_shape = inst_shape;
      } else {
        FOR_RANGE(int64_t, i, 1, dense_shape.NumAxes()) {
          CHECK_EQ(dense_shape.At(i), inst_shape.At(i));
        }
        dense_shape.Set(0, dense_shape.At(0) + inst_shape.At(0));
      }
    }
    blob->tree_lod_mut_view().UpdateLoD(lod_tree);
  }
  auto* dense_shape_mut_view = blob->dense_shape_mut_view();
  if (dense_shape_mut_view) { dense_shape_mut_view->set_shape(dense_shape); }
}

REGISTER_KERNEL(OperatorConf::kDataLoadConf, DataLoadKernel);

}  // namespace oneflow
