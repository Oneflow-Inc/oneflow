#ifndef ONEFLOW_CUSTOMIZED_SUMMARY_EVENTS_WRITER_H_
#define ONEFLOW_CUSTOMIZED_SUMMARY_EVENTS_WRITER_H_

#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/common/util.h"
#include "oneflow/customized/summary/crc32c.h"
#include "oneflow/customized/summary/event.pb.h"

#include <time.h>
#include <mutex>

namespace oneflow {

namespace summary {

#define MAX_QUEUE_NUM 10
#define FLUSH_TIME 3 * 60 * 1000 * 1000
#define FILE_VERSION "brain.Event:3"
const size_t kHeadSize = sizeof(uint64_t) + sizeof(uint32_t);
const size_t kTailSize = sizeof(uint32_t);

class EventsWriter {
 public:
  EventsWriter();
  ~EventsWriter();

  Maybe<void> Init(const std::string& logdir);
  void WriteEvent(const Event& event);
  void Flush();
  Maybe<void> Close();

  void AppendQueue(std::unique_ptr<Event> event);
  Maybe<void> FileFlush();

 private:
  Maybe<void> TryToInit();

  inline static uint32_t MaskedCrc(const char* data, size_t size) {
    return crc32c::Mask(crc32c::Value(data, size));
  }

  inline static void EncodeHead(char* head, const char* data, size_t size);
  inline static void EncodeTail(char* tail, const char* data, size_t size);

  std::string log_dir_;
  std::string filename_;
  std::unique_ptr<fs::FileSystem> file_system_;
  std::unique_ptr<fs::WritableFile> writable_file_;
  uint64_t last_flush_time_;
  std::vector<std::unique_ptr<Event>> event_queue_;
  std::mutex queue_mutex;
  OF_DISALLOW_COPY(EventsWriter);
};

void EventsWriter::EncodeHead(char* head, const char* data, size_t size) {
  crc32c::EncodeFixed64(head + 0, size);
  crc32c::EncodeFixed32(head + sizeof(uint64_t), MaskedCrc(head, sizeof(uint64_t)));
}

void EventsWriter::EncodeTail(char* tail, const char* data, size_t size) {
  crc32c::EncodeFixed32(tail, MaskedCrc(data, size));
}

}  // namespace summary

}  // namespace oneflow

#endif
