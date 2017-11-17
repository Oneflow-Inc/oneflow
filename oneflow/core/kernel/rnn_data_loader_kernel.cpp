#include "oneflow/core/kernel/rnn_data_loader_kernel.h"
#include <algorithm>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

template<typename T>
void RnnDataLoaderKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto ctx = static_cast<std::pair<DataLoadBuf, int64_t>*>(kernel_ctx.other);
  DataLoadBuf& dl_buf = ctx->first;
  InitInStream(ctx->second);

  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(GetDataType<T>::val, out_blob->data_type());

  kernel_ctx.device_ctx->cpu_stream()->SendWork([out_blob, &dl_buf, this]) {
    if (dl_buf.max_real_col_num == 0) {
      if (dl_buf.row_num == 0) {    // first time, initialize
        dl_buf.row_num = op()->GetInt64FromSpecialConf("piece_size");
        dl_buf.col_num = op()->GetInt64FromSpecialConf("max_col_size");
        dl_buf.piece.resize(dl_buf.row_num * dl_buf.col_num);
        dl_buf.data_ids.resize(dl_buf.row_num);
      } 
      
      // read from in_stream
      int64_t piece_size = dl_buf.row_num;
      std::string line;
      std::string token;
      for (size_t i = 0;i < piece_size; ++i) {
        int32_t read_status = in_stream->ReadLine(&line);
        if (read_status == 0) {
          const char* line_ptr = line.c_str();
          line_ptr = StrToToken(line_ptr, ",", &token) + 1;
          if (out_blob->has_data_id()) {
            CHECK_LE(token.size(), JobDesc::Singleton()->SizeOfOneDataID());
            dl_buf.data_ids.at(i) = std::move(token);  // TODO: token becomes unspecific state, no side effects?
          }
          int64_t cnt = 0;
          while(*(line_ptr - 1) != '\0') {
            line_ptr = StrToToken(line_ptr, ",", &token) + 1;
            dl_buf.piece.at(i * piece_size + cnt) = static_case<int64_t>(oneflow_cast<T>(token));
            cnt ++;
          }
          CHECK_LE(cnt, dl_buf.col_num);
          dl_buf.max_real_col_num = std::max(dl_buf.max_real_col_num, cnt);
        } else {
          CHECK_EQ(-1, read_status);
          CHECK(out_blob->has_data_id());   //TODO: why?
        }
      }
    }

    // write into out_blob from dl_conf
    CHECK_EQ(dl_buf.row_num, out_blob->shape().At(0));
    CHECK_EQ(1, out_blob->shape().At(1));

    int64_t piece_size = dl_buf.row_num;
    T* out_dptr = out_blob->mut_dptr<T>();
    for (int64_t i = 0;i < piece_size; ++i) {
      if (out_blob->has_data_id()) {
        const std::string& cur_data_id = dl_buf.data_ids.at(i);
        memcpy(out_blob->mut_data_id(i), cur_data_ids.c_str(), cur_data_ids.size());
        if (cur_data_ids.size() != JobDesc::Singleton()->SizeOfOneDataID()) {
          *(out_blob->mut_data_id(i) + cur_data_ids.size()) = '\0';
        }
      }
      *out_dptr = oneflow_cast<T>(
          dl_buf.piece.at(i * dl_buf.col_num + dl_buf.col_id));
      out_dptr ++;
      dl_buf.col_id ++;
      if (dl_buf.col_id == dl_buf.max_real_col_num) {
        dl_buf.max_real_col_num = 0;
        dl_buf.col_id = 0;
      }

    }
  }
}

template<typename T>
void RnnDataLoaderKernel<T>::InitInStream(int64_t parallel_id) const {
  if (in_stream_) { return; }
  std::string data_dir = op()->GetStringFromSpecialConf("data_dir");
  std::string file_path = data_dir + "part-" + std::to_string(parallel_id);
  if (JobDesc::Singleton()->is_train()) {
    in_stream_.reset(new CyclicPersistentInStream(GlobalFS(), file_path));
  } else {
    in_stream_.reset(new NormalPersistentInStream(GlobalFS(), file_path));
  }
}

} // namespace oneflow
