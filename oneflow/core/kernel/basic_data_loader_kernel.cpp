#include "oneflow/core/kernel/basic_data_loader_kernel.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

static int32_t next_col = 0;
static int32_t max_length = 0;
static int64_t piece_id = -1; 

template<typename T>
void BasicDataLoaderKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* buffer_blob = BnInOp2Blob("buffer");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(GetDataType<T>::val, out_blob->data_type());
  int64_t piece_size = out_blob->shape().At(0);
  int64_t max_seq_len = op_conf().basic_data_loader_conf().max_seq_len();
  T* out_dptr = out_blob->mut_dptr<T>();
  T* buffer_dptr = buffer_blob->mut_dptr<T>();

  // read data to buffer
  if (next_col >= max_length) {
    piece_id++;
    T* buffer_dptr = buffer_blob->mut_dptr<T>();
    std::string line;
    std::string token;
    // init the recorders
    next_col = 0;
    max_length = 0;
    // each line format:
    // (data_id)?(,data_content)*
    FOR_RANGE(int64_t, i, 0, piece_size) {
      T* each_buff_dptr = buffer_dptr + i * max_seq_len;
      // mention that the type of seq_len should be same with seq_len in blob
      int32_t seq_len = 0;
      int32_t read_status = in_stream_->ReadLine(&line);
      if (read_status == 0) {
        const char* line_ptr = line.c_str();
        line_ptr = StrToToken(line_ptr, ",", &token) + 1;
        // see if has data_id and set it
        if (buffer_blob->has_data_id()) {
          CHECK_LE(token.size(), JobDesc::Singleton()->SizeOfOneDataId());
          memcpy(buffer_blob->mut_data_id(i), token.c_str(), token.size());
          if (token.size() != JobDesc::Singleton()->SizeOfOneDataId()) {
            *(buffer_blob->mut_data_id(i) + token.size()) = '\0';
          }
        }
        // read and count the seq_len
        FOR_RANGE(int64_t, j, 0, buffer_blob->shape().At(1)) {
          // buffer data content in each column
          FOR_RANGE(int64_t, k, 0, buffer_blob->shape().Count(2)) {
            line_ptr = StrToToken(line_ptr, ",", &token) + 1;
            *each_buff_dptr++ = oneflow_cast<T>(token);
          }
          seq_len++;
          if (*(line_ptr - 1) == '\0') { break; }
        }
        // set seq_len
        buffer_blob->mut_seq_len(i) = seq_len;
        max_length = max_length > seq_len ? max_length : seq_len;
        CHECK_EQ(*(line_ptr - 1), '\0');
      } else {
        // add all 0 data to make piece full
        CHECK(kernel_ctx.other);
        *(static_cast<bool*>(kernel_ctx.other)) = true;
        CHECK_EQ(read_status, -1);
        if (buffer_blob->has_data_id()) {
          memset(buffer_blob->mut_data_id(i), '\0',
                 JobDesc::Singleton()->SizeOfOneDataId());
        }
        buffer_blob->mut_seq_len(i) = 0;
        FOR_RANGE(int64_t, j, 0, buffer_blob->shape().Count(1)) {
          *each_buff_dptr++ = static_cast<T>(0);
        }
      }
    }
  }

  // use data in buffer
  FOR_RANGE(int64_t, i, 0, piece_size) {
    // blob_header
    out_blob->mut_blob_header()->set_piece_id(piece_id);
    out_blob->mut_blob_header()->set_col_id(next_col);
    out_blob->mut_blob_header()->set_max_col_num(max_length);
    // data_id
    if (out_blob->has_data_id()) {
      memcpy(out_blob->mut_data_id(i), buffer_blob->data_id(i),
             JobDesc::Singleton()->SizeOfOneDataId());
    }
    // seq_len
    out_blob->mut_seq_len(i) = buffer_blob->seq_len(i);
    // data_content
    T* each_out_dptr = out_dptr + i * out_blob->shape().Count(1);
    T* each_buff_dptr = buffer_dptr + i * max_seq_len
                        + next_col * out_blob->shape().Count(1);
    if (next_col < buffer_blob->seq_len(i)) {
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

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicDataLoaderConf,
                               BasicDataLoaderKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
