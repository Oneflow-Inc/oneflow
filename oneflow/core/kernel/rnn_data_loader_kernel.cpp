#include "oneflow/core/kernel/rnn_data_loader_kernel.h"
#include <algorithm>
#include "oneflow/core/actor/rnn_source_compute_actor.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

template<typename IntegerT>
void RnnDataLoaderKernel<IntegerT>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK_NOTNULL(kernel_ctx.other);
  auto ctx = static_cast<std::pair<RnnSourceCompActor::DataLoadBuf*, bool*>*>(
      kernel_ctx.other);
  RnnSourceCompActor::DataLoadBuf* dl_buf = ctx->first;

  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(GetDataType<IntegerT>::val, out_blob->data_type());

  // read from instream
  if (dl_buf->max_real_col_num == 0) {
    if (dl_buf->row_num == 0) {
      CHECK(op_conf().has_rnn_data_loader_kernel());
      dl_buf->row_num = op_conf().rnn_data_loader_conf().piece_size();
      dl_buf->col_num = op_conf().rnn_data_loader_conf().max_col_size();
      dl_buf->piece.resize(dl_buf->row_num * dl_buf->col_num, -1);
      dl_buf->data_ids.resize(dl_buf->row_num);
      dl_buf->offsets.resize(dl_buf->row_num);
    }

    int64_t piece_size = dl_buf->row_num;
    std::string line;
    std::string token;
    for (int64_t i = 0; i < piece_size; ++i) {
      int32_t read_status = in_stream_->ReadLine(&line);
      if (read_status == 0) {
        const char* line_ptr = line.c_str();
        line_ptr = StrToToken(line_ptr, ",", &token) + 1;
        if (out_blob->has_data_id()) {
          CHECK_LE(token.size(), JobDesc::Singleton()->SizeOfOneDataId());
          dl_buf->data_ids.at(i) = std::move(token);
        }
        int64_t cnt = 0;
        while (*(line_ptr - 1) != '\0') {
          line_ptr = StrToToken(line_ptr, ",", &token) + 1;
          dl_buf->piece.at(i * piece_size + cnt) =
              static_cast<int64_t>(oneflow_cast<IntegerT>(token));
          cnt++;
        }
        dl_buf->offsets.at(i) = cnt;
        CHECK_LE(dl_buf->offsets.at(i), dl_buf->col_num);
        dl_buf->max_real_col_num =
            std::max(dl_buf->max_real_col_num, dl_buf->offsets.at(i));
      } else {
        CHECK_EQ(read_status, -1);
        *(ctx->second) = true;
        CHECK(out_blob->has_data_id());
        break;
      }
    }
  }

  // write into out_blob from dl_conf
  CHECK_EQ(dl_buf->row_num, out_blob->shape().At(0));
  CHECK_EQ(1, out_blob->shape().At(1));
  IntegerT* out_dptr = out_blob->mut_dptr<IntegerT>();
  for (int64_t i = 0; i < dl_buf->row_num; ++i) {
    if (out_blob->has_data_id()) {
      const std::string& cur_data_id = dl_buf->data_ids.at(i);
      memcpy(out_blob->mut_data_id(i), cur_data_id.c_str(), cur_data_id.size());
      if (cur_data_id.size() < JobDesc::Singleton()->SizeOfOneDataId()) {
        *(out_blob->mut_data_id(i) + cur_data_id.size()) = '\0';
      }
    }

    CHECK(out_blob->has_offset());
    out_blob->mut_offset(i) = dl_buf->offsets.at(i);

    *out_dptr = static_cast<IntegerT>(
        dl_buf->piece.at(i * dl_buf->col_num + dl_buf->col_id));
    out_dptr++;
  }
  dl_buf->col_id++;
  if (dl_buf->col_id == dl_buf->max_real_col_num) {
    dl_buf = RnnSourceCompActor::DataLoadBuf();
  }
}

template<typename IntegerT>
void RnnDataLoaderKernel<IntegerT>::VirtualKernelInit(
    const ParallelContext* parallel_ctx) {
  const std::string& data_dir = op_conf().rnn_data_loader_conf().data_dir();
  std::string parallel_id = std::to_string(parallel_ctx->parallel_id());
  std::string file_path = JoinPath(data_dir, "part-" + parallel_id);
  if (JobDesc::Singleton()->IsTrain()) {
    in_stream_.reset(new CyclicPersistentInStream(GlobalFS(), file_path));
  } else {
    in_stream_.reset(new NormalPersistentInStream(GlobalFS(), file_path));
  }
}

}  // namespace oneflow
