#include "oneflow/core/kernel/basic_data_loader_kernel.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/actor/source_compute_actor.h"

namespace oneflow {

template<typename T>
void BasicDataLoaderKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto status = static_cast<SourceCompActor::DataLoadStatus*>(kernel_ctx.other);
  CHECK_EQ(GetDataType<T>::val, out_blob->data_type());

  if (out_blob->has_seq_len()) {
    Blob* buffer_blob = BnInOp2Blob("buffer");
    CHECK(buffer_blob);
    CHECK_EQ(GetDataType<T>::val, buffer_blob->data_type());
    CHECK(kernel_ctx.other);
    // read data to buffer
    if (status->next_col_id >= status->max_col_num) {
      status->piece_id++;
      status->next_col_id = 0;
      ReadOnePieceToBuffer(kernel_ctx, buffer_blob);
    }
    ReadBufferToOutBlob(kernel_ctx, buffer_blob, out_blob);
    status->next_col_id++;
  } else {
    status->piece_id++;
    ReadDirectToOutBlob(kernel_ctx, out_blob);
  }
}

template<typename T>
void BasicDataLoaderKernel<T>::VirtualKernelInit(
    const ParallelContext* parallel_ctx) {
  const std::string& data_dir = op_conf().basic_data_loader_conf().data_dir();
  std::string parallel_id = std::to_string(parallel_ctx->parallel_id());
  std::string file_path = JoinPath(data_dir, "part-" + parallel_id);
  if (JobDesc::Singleton()->IsTrain()) {
    in_stream_.reset(new CyclicPersistentInStream(GlobalFS(), file_path));
  } else {
    in_stream_.reset(new NormalPersistentInStream(GlobalFS(), file_path));
  }
}

template<typename T>
void BasicDataLoaderKernel<T>::ReadDirectToOutBlob(const KernelCtx& kernel_ctx,
                                                   Blob* out_blob) const {
  CHECK(!out_blob->has_seq_len());
  CHECK_EQ(GetDataType<T>::val, out_blob->data_type());
  int64_t piece_size = out_blob->shape().At(0);
  T* out_dptr = out_blob->mut_dptr<T>();
  std::string line;
  std::string token;
  FOR_RANGE(int64_t, i, 0, piece_size) {
    int32_t read_status = in_stream_->ReadLine(&line);
    if (read_status == 0) {
      const char* line_ptr = line.c_str();
      line_ptr = StrToToken(line_ptr, ",", &token) + 1;
      if (out_blob->has_data_id_field()) {
        CHECK_LE(token.size(), JobDesc::Singleton()->SizeOfOneDataId());
        memcpy(out_blob->mut_data_id(i), token.c_str(), token.size());
        if (token.size() != JobDesc::Singleton()->SizeOfOneDataId()) {
          *(out_blob->mut_data_id(i) + token.size()) = '\0';
        }
      }
      FOR_RANGE(int64_t, j, 0, out_blob->shape().Count(1)) {
        line_ptr = StrToToken(line_ptr, ",", &token) + 1;
        *out_dptr++ = oneflow_cast<T>(token);
      }
      CHECK_EQ(*(line_ptr - 1), '\0');
    } else {
      LOG(INFO)<<"end";
      CHECK(kernel_ctx.other);
      auto status =
          static_cast<SourceCompActor::DataLoadStatus*>(kernel_ctx.other);
      status->is_eof = true;
      CHECK_EQ(read_status, -1);
      FillBlobRowsWithZero(out_blob, i, piece_size);
      break;
    }
  }
}

template<typename T>
void BasicDataLoaderKernel<T>::ReadOnePieceToBuffer(const KernelCtx& kernel_ctx,
                                                    Blob* buffer_blob) const {
  CHECK(buffer_blob->has_seq_len());
  CHECK(kernel_ctx.other);
  auto status = static_cast<SourceCompActor::DataLoadStatus*>(kernel_ctx.other);
  int64_t piece_size = buffer_blob->shape().At(0);
  int64_t max_seq_len = op_conf().basic_data_loader_conf().max_seq_len();
  status->max_col_num = 0;
  T* buffer_dptr = buffer_blob->mut_dptr<T>();

  std::string line;
  std::string token;

  // each line format: (data_id)?(,data_content)*
  FOR_RANGE(int64_t, i, 0, piece_size) {
    T* each_buff_dptr = buffer_dptr + i * max_seq_len;
    int32_t seq_len = 0;
    int32_t read_status = in_stream_->ReadLine(&line);
    if (read_status == 0) {
      const char* line_ptr = line.c_str();
      line_ptr = StrToToken(line_ptr, ",", &token) + 1;
      if (buffer_blob->has_data_id()) {
        CHECK_LE(token.size(), JobDesc::Singleton()->SizeOfOneDataId());
        memcpy(buffer_blob->mut_data_id(i), token.c_str(), token.size());
        if (token.size() != JobDesc::Singleton()->SizeOfOneDataId()) {
          *(buffer_blob->mut_data_id(i) + token.size()) = '\0';
        }
      }
      FOR_RANGE(int64_t, j, 0, buffer_blob->shape().At(1)) {
        FOR_RANGE(int64_t, k, 0, buffer_blob->shape().Count(2)) {
          line_ptr = StrToToken(line_ptr, ",", &token) + 1;
          *each_buff_dptr++ = oneflow_cast<T>(token);
        }
        seq_len++;
        if (*(line_ptr - 1) == '\0') { break; }
      }
      *(buffer_blob->mut_seq_len(i)) = seq_len;
      status->max_col_num = status->max_col_num > status->max_col_num
                                ? status->max_col_num
                                : seq_len;
      CHECK_EQ(*(line_ptr - 1), '\0');
    } else {
      status->is_eof = true;
      CHECK_EQ(read_status, -1);
      FillBlobRowsWithZero(buffer_blob, i, piece_size);
      break;
    }
  }
}

template<typename T>
void BasicDataLoaderKernel<T>::ReadBufferToOutBlob(const KernelCtx& kernel_ctx,
                                                   Blob* buffer_blob,
                                                   Blob* out_blob) const {
  CHECK(out_blob->has_seq_len());
  CHECK(kernel_ctx.other);
  auto status = static_cast<SourceCompActor::DataLoadStatus*>(kernel_ctx.other);
  T* out_dptr = out_blob->mut_dptr<T>();
  T* buffer_dptr = buffer_blob->mut_dptr<T>();
  int64_t piece_size = out_blob->shape().At(0);
  int64_t max_seq_len = op_conf().basic_data_loader_conf().max_seq_len();

  out_blob->set_piece_id(status->piece_id);
  out_blob->set_max_col_num(status->max_col_num);
  out_blob->set_col_id(status->next_col_id);

  if (out_blob->has_data_id()) {
    memcpy(out_blob->mut_data_id(), buffer_blob->data_id(),
           out_blob->ByteSizeOfDataIdField());
  }

  FOR_RANGE(int64_t, i, 0, piece_size) {
    T* each_out_dptr = out_dptr + i * out_blob->shape().Count(1);
    T* each_buff_dptr = buffer_dptr + i * max_seq_len
                        + status->next_col_id * out_blob->shape().Count(1);
    *(out_blob->mut_seq_len(i)) = buffer_blob->seq_len(i);
    if (status->next_col_id < buffer_blob->seq_len(i)) {
      FOR_RANGE(int64_t, j, 0, out_blob->shape().Count(1)) {
        *each_out_dptr++ = *each_buff_dptr++;
      }
    } else {
      FOR_RANGE(int64_t, j, 0, out_blob->shape().Count(1)) {
        *each_out_dptr++ = static_cast<T>(0);
      }
    }
  }
}

template<typename T>
void BasicDataLoaderKernel<T>::FillBlobRowsWithZero(Blob* blob, int64_t start,
                                                    int64_t end) const {
  if (blob->has_data_id()) {
    memset(blob->mut_data_id(start), '\0',
           (end - start) * JobDesc::Singleton()->SizeOfOneDataId());
  }
  if (blob->has_seq_len()) {
    FOR_RANGE(int64_t, i, start, end) { *(blob->mut_seq_len(i)) = 0; }
  }
  FOR_RANGE(int64_t, i, start, end) {
    T* dptr = blob->mut_dptr<T>() + i * blob->shape().Count(1);
    FOR_RANGE(int64_t, j, 0, blob->shape().Count(1)) {
      *dptr++ = static_cast<T>(0);
    }
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicDataLoaderConf,
                               BasicDataLoaderKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
