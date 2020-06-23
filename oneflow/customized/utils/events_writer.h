#ifndef ONEFLOW_UTIL_EVENTS_WRITER_H_
#define ONEFLOW_UTIL_EVENTS_WRITER_H_

#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/event.pb.h"
#include "oneflow/customized/utils/crc32c.h"

#include <sys/time.h>
#include <time.h>

namespace oneflow {

class EventsWriter {
 public:
  explicit EventsWriter(const std::string& file_prefix);
  ~EventsWriter();

  static const size_t kHeaderSize = sizeof(uint64_t) + sizeof(uint32_t);
  static const size_t kFooterSize = sizeof(uint32_t);

  Maybe<void> Initialize(const std::string& logdir, const std::string& filename_suffix);
  Maybe<void> Init();
  Maybe<void> InitWithSuffix(const std::string& suffix);

  std::string FileName();
  void WriteEvent(const Event& event);
  Maybe<void> WriteRecord(std::string data);
  void WriteSerializedEvent(std::string event_str);
  Maybe<void> Flush();
  Maybe<void> Close();

  void AppendQueue(std::unique_ptr<Event> event);
  void InternalFlush();

  inline static void PopulateHeader(char* header, const char* data, size_t n);
  inline static void PopulateFooter(char* footer, const char* data, size_t n);

 private:
  Maybe<void> InitIfNeeded();
  Maybe<void> FileStillExists();

  inline static uint32_t MaskedCrc(const char* data, size_t n) {
    return crc32c::Mask(crc32c::Value(data, n));
  }

  const std::string file_prefix_;
  std::string log_dir_;
  std::string file_suffix_;
  std::string filename_;
  std::unique_ptr<fs::FileSystem> file_system_;
  std::unique_ptr<fs::WritableFile> writable_file_;
  const int max_queue_;
  const int flush_millis_;
  uint64_t last_flush_;
  std::vector<std::unique_ptr<Event>> queue_;
  // int num_outstanding_events_;
  OF_DISALLOW_COPY(EventsWriter);
};

void EventsWriter::PopulateHeader(char* header, const char* data, size_t n) {
  crc32c::EncodeFixed64(header + 0, n);
  crc32c::EncodeFixed32(header + sizeof(uint64_t), MaskedCrc(header, sizeof(uint64_t)));
}

void EventsWriter::PopulateFooter(char* footer, const char* data, size_t n) {
  crc32c::EncodeFixed32(footer, MaskedCrc(data, n));
}

}  // namespace oneflow

#endif
