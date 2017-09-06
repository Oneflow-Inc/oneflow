#include "oneflow/core/persistence/cyclic_data_reader.h"

namespace oneflow {

CyclicDataReader::CyclicDataReader(const std::string& file_path) {
  tensorflow::Env* env = tensorflow::Env::Default();
  TF_CHECK_OK(env->NewRandomAccessFile(file_path, &file_));
  in_.reset(new tensorflow::io::InputBuffer(file_.get(), 64 * 1024 * 1024));
}

int32_t CyclicDataReader::ReadLine(std::string* line) {
  tensorflow::Status status = in_->ReadLine(line);
  if (status.code() == tensorflow::error::OUT_OF_RANGE) {
    TF_CHECK_OK(in_->Seek(0));
    status = in_->ReadLine(line);
  }
  TF_CHECK_OK(status);
  return 0;
}

}  // namespace oneflow
