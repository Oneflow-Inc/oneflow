#include "oneflow/core/kernel/basic_data_loader_kernel.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

template<typename T>
void BasicDataLoaderKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(kernel_ctx.other);
  auto status = static_cast<SourceCompActor::DataLoadStatus*>(kernel_ctx.other);
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(GetDataType<T>::val, out_blob->data_type());

  if (out_blob->max_col_num() == 1) {
    Blob* buffer_blob = BnInOp2Blob("buffer");
    CHECK_EQ(GetDataType<T>::val, buffer_blob->data_type());
    if (status->next_col_id > status->max_col_id) {
      ReadOnePieceToBlob(status, buffer_blob);
    }
    ReadOneColFromBufferToOutBlob(kernel_ctx, status, buffer_blob, out_blob);
  } else {
    ReadOnePieceToBlob(status, out_blob);
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
void BasicDataLoaderKernel<T>::ReadOnePieceToBlob(
    SourceCompActor::DataLoadStatus* status, Blob* blob) const {
  status->max_col_id = -1;
  status->next_col_id = 0;
  std::string line;
  blob->set_col_id(0);
  blob->set_max_col_id(0);
  FOR_RANGE(int64_t, i, 0, blob->shape().At(0)) {
    int32_t read_status = in_stream_->ReadLine(&line);
    if (read_status == 0) {
      const char* line_ptr = line.c_str();
      line_ptr = ReadOneDataId(line_ptr, blob, i);
      int32_t max_col_id_of_this_line = ReadOneDataContent(line_ptr, blob, i);
      if (blob->has_col_num_field()) {
        blob->set_col_num(i, max_col_id_of_this_line + 1);
        if (status->max_col_id < max_col_id_of_this_line) {
          status->max_col_id = max_col_id_of_this_line;
        }
      }
    } else {
      CHECK_EQ(read_status, -1);
      status->is_eof = true;
      if (blob->has_data_id_field()) {
        memset(blob->mut_data_id(i), '\0',
               (blob->shape().At(0) - i)
                   * JobDesc::Singleton()->SizeOfOneDataId());
      }
      if (blob->has_col_num_field()) {
        FOR_RANGE(int64_t, j, i, blob->shape().At(0)) {
          blob->set_col_num(i, 0);
        }
      }
      T* dptr = blob->mut_dptr<T>() + i * blob->shape().Count(1);
      memset(dptr, static_cast<T>(0),
             (blob->shape().Count(0) - i * blob->shape().Count(1)));
      break;
    }
  }
  status->next_piece_id += 1;
}

template<typename T>
void BasicDataLoaderKernel<T>::ReadOneColFromBufferToOutBlob(
    const KernelCtx& kernel_ctx, SourceCompActor::DataLoadStatus* status, 
    const Blob* buffer_blob, Blob* out_blob) const {
  out_blob->set_max_col_id(status->max_col_id);
  out_blob->set_col_id(status->next_col_id);
  if (out_blob->has_data_id_field()) {
    out_blob->CopyDataIdFrom<DeviceType::kCPU>(kernel_ctx.device_ctx,
                                               buffer_blob);
  }
  if (out_blob->has_col_num_field()) {
    out_blob->CopyColNumFrom<DeviceType::kCPU>(kernel_ctx.device_ctx,
                                               buffer_blob);
  }
  FOR_RANGE(int64_t, i, 0, out_blob->shape().At(0)) {
    T* each_out_dptr = out_blob->mut_dptr<T>() + i * out_blob->shape().Count(1);
    const T* each_buff_dptr =
        buffer_blob->dptr<T>() + i * buffer_blob->shape().Count(1)
        + status->next_col_id * buffer_blob->shape().Count(2);
    memcpy(each_out_dptr, each_buff_dptr, out_blob->shape().Count(1));
  }
  status->next_col_id += 1;
}

template<typename T>
const char* BasicDataLoaderKernel<T>::ReadOneDataId(const char* line_ptr,
                                                    Blob* blob,
                                                    int64_t index) const {
  std::string token;
  line_ptr = StrToToken(line_ptr, ",", &token) + 1;
  if (blob->has_data_id_field()) {
    CHECK_LE(token.size(), JobDesc::Singleton()->SizeOfOneDataId());
    memcpy(blob->mut_data_id(index), token.c_str(), token.size());
    if (token.size() != JobDesc::Singleton()->SizeOfOneDataId()) {
      *(blob->mut_data_id(index) + token.size()) = '\0';
    }
  }
  return line_ptr;
}

template<typename T>
int32_t BasicDataLoaderKernel<T>::ReadOneDataContent(const char* line_ptr,
                                                     Blob* blob,
                                                     int64_t index) const {
  std::string token;
  int32_t each_max_col_id = -1;
  T* each_dptr = blob->mut_dptr<T>() + index * blob->shape().Count(1);
  if (blob->has_col_num_field()) {
    FOR_RANGE(int64_t, j, 0, blob->shape().At(1)) {
      FOR_RANGE(int64_t, k, 0, blob->shape().Count(2)) {
        line_ptr = StrToToken(line_ptr, ",", &token) + 1;
        *each_dptr++ = oneflow_cast<T>(token);
      }
      ++each_max_col_id;
      if (*(line_ptr - 1) == '\0') { break; }
    }
  } else {
    FOR_RANGE(int64_t, j, 0, blob->shape().Count(1)) {
      line_ptr = StrToToken(line_ptr, ",", &token) + 1;
      *each_dptr++ = oneflow_cast<T>(token);
    }
  }
  CHECK_EQ(*(line_ptr - 1), '\0');
  return each_max_col_id;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicDataLoaderConf,
                               BasicDataLoaderKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
