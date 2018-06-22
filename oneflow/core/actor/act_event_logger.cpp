#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/common/protobuf.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

const std::string ActEventLogger::experiment_prefix_("experiment_");
const std::string ActEventLogger::act_event_bin_filename_("act_event.bin");
const std::string ActEventLogger::act_event_txt_filename_("act_event.txt");

void ActEventLogger::PrintActEventsToLogDir(const ActEvents& act_events) {
  int64_t act_event_num = act_events.act_events_size();
  FOR_RANGE(int64_t, i, 0, act_event_num) {
    auto& act_event = act_events.act_events(i);
    bin_out_stream_ << act_event;
    std::string act_event_txt;
    google::protobuf::TextFormat::PrintToString(act_event, &act_event_txt);
    txt_out_stream_ << act_event_txt;
  }
}

ActEventLogger::ActEventLogger(bool is_experiment)
    : bin_out_stream_(LocalFS(), JoinPath(LogDir(), (is_experiment ? experiment_prefix_ : "")
                                                        + act_event_bin_filename_)),
      txt_out_stream_(LocalFS(), JoinPath(LogDir(), (is_experiment ? experiment_prefix_ : "")
                                                        + act_event_txt_filename_)) {}

void ParseActEvents(const std::string& act_event_filepath, std::list<ActEvent>* act_events) {
  NormalPersistentInStream in_stream(LocalFS(), act_event_filepath);
  int64_t act_event_size;
  while (!in_stream.Read(reinterpret_cast<char*>(&act_event_size), sizeof(act_event_size))) {
    std::vector<char> buffer(act_event_size);
    CHECK(!in_stream.Read(buffer.data(), act_event_size));
    act_events->emplace_back();
    act_events->back().ParseFromArray(buffer.data(), act_event_size);
  }
}
}  // namespace oneflow
