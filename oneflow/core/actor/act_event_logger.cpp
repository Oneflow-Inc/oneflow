#include "oneflow/core/actor/act_event_logger.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

void ActEventLogger::PrintActEventToLogDir(const ActEvent& act_event) {
  std::string act_event_bin = act_event.SerializeAsString();
  size_t act_event_bin_size = act_event_bin.size();
  bin_out_stream_ << act_event_bin_size << act_event_bin;
  std::string act_event_txt;
  google::protobuf::TextFormat::PrintToString(act_event, &act_event_txt);
  txt_out_stream_ << act_event_txt;
}

ActEventLogger::ActEventLogger()
    : bin_out_stream_(LocalFS(), LogDir() + "/act_event.bin"),
      txt_out_stream_(LocalFS(), LogDir() + "/act_event.txt") {}

}  // namespace oneflow
