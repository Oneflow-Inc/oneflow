#include "oneflow/core/persistence/cyclic_data_reader.h"

namespace oneflow {

const size_t CyclicDataReader::buffer_size_ = 128 * 1024 * 1024;

CyclicDataReader::CyclicDataReader(const std::string& file_path) {
  FS_CHECK_OK(GlobalFS()->NewRandomAccessFile(file_path, &file_));
  FS_CHECK_OK(GlobalFS()->GetFileSize(file_path, &file_size_));
  cur_file_pos_ = 0;
  buffer_ = new char[buffer_size_ + 1];
  UpdateBuffer();
}

CyclicDataReader::~CyclicDataReader() { delete[] buffer_; }

int32_t CyclicDataReader::ReadLine(std::string* line) {
  if (cur_file_pos_ == file_size_) { return -1; }
  line->clear();
  while (*cur_buf_begin_ != '\n') {
    if (cur_buf_begin_ == cur_buf_end_) {
      UpdateBuffer();
      if (cur_buf_begin_ == cur_buf_end_) {
        return 0;
      } else {
        continue;
      }
    }
    line->push_back(*cur_buf_begin_++);
  }
  ++cur_buf_begin_;
  return 0;
}

void CyclicDataReader::UpdateBuffer() {
  size_t n = std::min(buffer_size_, file_size_ - cur_file_pos_);
  if (n == 0) { return; }
  file_->Read(cur_file_pos_, n, buffer_);
  cur_file_pos_ += n;
  cur_buf_begin_ = buffer_;
  cur_buf_end_ = buffer_ + n;
  *cur_buf_end_ = '\0';
  if (cur_file_pos_ == file_size_) { cur_file_pos_ = 0; }
}

}  // namespace oneflow
