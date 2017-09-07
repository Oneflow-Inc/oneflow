#include "oneflow/core/persistence/data_reader.h"

namespace oneflow {

const uint64_t DataReader::buffer_size_ = 128 * 1024 * 1024;

DataReader::~DataReader() { delete[] buffer_; }

int32_t DataReader::ReadLine(std::string* line) {
  if (IsEof()) { return -1; }
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

int32_t DataReader::Read(char* s, size_t n) {
  if (IsEof()) { return -1; }
  while (n--) {
    if (cur_buf_begin_ == cur_buf_end_) { UpdateBuffer(); }
    CHECK_NE(cur_buf_begin_, cur_buf_end_);
    *s++ = *cur_buf_begin_++;
  }
  return 0;
}

DataReader::DataReader(fs::FileSystem* fs, const std::string& file_path) {
  FS_CHECK_OK(fs->NewRandomAccessFile(file_path, &file_));
  FS_CHECK_OK(fs->GetFileSize(file_path, &file_size_));
  cur_file_pos_ = 0;
  buffer_ = new char[buffer_size_ + 1];
  cur_buf_begin_ = buffer_;
  cur_buf_end_ = buffer_;
  *cur_buf_end_ = '\0';
}

bool DataReader::IsEof() const {
  return cur_buf_begin_ == cur_buf_end_ && cur_file_pos_ == file_size_;
}

void DataReader::UpdateBuffer() {
  CHECK_EQ(cur_buf_begin_, cur_buf_end_);
  uint64_t n = std::min(buffer_size_, file_size_ - cur_file_pos_);
  if (n == 0) { return; }
  file_->Read(cur_file_pos_, n, buffer_);
  cur_buf_begin_ = buffer_;
  cur_buf_end_ = buffer_ + n;
  *cur_buf_end_ = '\0';
  AddNForCurFilePos(n);
}

}  // namespace oneflow
