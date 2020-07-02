#ifndef ONEFLOW_UTIL_EVENTS_WRITER_H_
#define ONEFLOW_UTIL_EVENTS_WRITER_H_

#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/event.pb.h"
#include "oneflow/customized/utils/crc32c.h"

#include <sys/time.h>
#include <time.h>
#include <mutex>

#define MAX_QUEUE_NUM 10
#define FLUSH_MILLIS 2 * 60 * 1000
#define FILE_VERSION "brain.Event:3"
const size_t kHeadSize = sizeof(uint64_t) + sizeof(uint32_t);
const size_t kTailSize = sizeof(uint32_t);

namespace oneflow {

class EventsWriter {
 public:
  EventsWriter();
  ~EventsWriter();

  Maybe<void> Init(const std::string& logdir);
  std::string FileName();
  void WriteEvent(const Event& event);
  Maybe<void> WriteRecord(const std::string& data);
  void WriteSerializedEvent(const std::string& event_str);
  Maybe<void> Flush();
  Maybe<void> Close();

  void AppendQueue(std::unique_ptr<Event> event);
  void InternalFlush();

 private:
  Maybe<void> TryToInit();

  inline static uint32_t MaskedCrc(const char* data, size_t n) {
    return crc32c::Mask(crc32c::Value(data, n));
  }

  inline static void EncodeHead(char* header, const char* data, size_t n);
  inline static void EncodeTail(char* footer, const char* data, size_t n);

  std::string log_dir_;
  std::string filename_;
  std::unique_ptr<fs::FileSystem> file_system_;
  std::unique_ptr<fs::WritableFile> writable_file_;
  const int max_queue_;
  const int flush_millis_;
  uint64_t last_flush_;
  std::vector<std::unique_ptr<Event>> queue_;
  // int num_outstanding_events_;
  std::mutex queue_mutex;
  OF_DISALLOW_COPY(EventsWriter);
};

void EventsWriter::EncodeHead(char* header, const char* data, size_t n) {
  crc32c::EncodeFixed64(header + 0, n);
  crc32c::EncodeFixed32(header + sizeof(uint64_t), MaskedCrc(header, sizeof(uint64_t)));
}

void EventsWriter::EncodeTail(char* footer, const char* data, size_t n) {
  crc32c::EncodeFixed32(footer, MaskedCrc(data, n));
}

}  // namespace oneflow

#endif
