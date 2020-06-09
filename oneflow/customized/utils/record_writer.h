#ifndef OF_CORE_LIB_IO_RECORD_WRITER_H_
#define OF_CORE_LIB_IO_RECORD_WRITER_H_

/*
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/persistence/file_system.h"
//#include "absl/strings/"
namespace oneflow {

//class WritableFile;

namespace io {

class RecordWriterOptions {
 public:
  enum CompressionType { NONE = 0, ZLIB_COMPRESSION = 1 };
  CompressionType compression_type = NONE;

  static RecordWriterOptions CreateRecordWriterOptions(const std::string& compression_type);
};

class RecordWriter {
 public:
  static const size_t kHeaderSize = sizeof(uint64_t) + sizeof(uint32_t);
  static const size_t kFooterSize = sizeof(uint32_t);

  RecordWriter(fs::WritableFile* dest, const RecordWriterOptions& options = RecordWriterOptions());

  ~RecordWriter();
  Maybe<void> WriteRecord(absl::string_view slice);

  Maybe<void> Flush();

  Maybe<void> Close();

  inline static void PopulateHeader(char* header, const char* data, size_t n);

  inline static void PopulateFooter(char* footer, const char* data, size_t n);

  private:
  fs::WritableFile* dest_;
  RecordWriterOptions options_;

  inline static uint32_t MaskedCrc(const char* data, size_t n) {
    return crc32c::Mask(crc32c::Value(data, n));
  }

  OF_DISALLOW_COPY(RecordWriterOptions);
};


void RecordWriter::PopulateHeader(char* header, const char* data, size_t n) {
  core::EncodeFixed64(header + 0, n);
  core::EncodeFixed32(header + sizeof(uint64),
                      MaskedCrc(header, sizeof(uint64)));
}

void RecordWriter::PopulateFooter(char* footer, const char* data, size_t n) {
  core::EncodeFixed32(footer, MaskedCrc(data, n));
}

}  // namespace io
}  // namespace oneflow

*/
#endif