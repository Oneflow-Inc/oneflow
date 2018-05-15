#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/common/protobuf.h"
#include <google/protobuf/text_format.h>

DEFINE_bool(force_flush_act_event, false, "flush act events forcely");

namespace oneflow {

const std::string ActEventLogger::act_event_bin_filename_("act_event.bin");
const std::string ActEventLogger::act_event_txt_filename_("act_event.txt");

void ActEventLogger::PrintActEventToLogDir(const ActEvent& act_event) {
  bin_out_stream_ << act_event;
  if (FLAGS_force_flush_act_event) { bin_out_stream_.Flush(); }
  std::string act_event_txt;
  google::protobuf::TextFormat::PrintToString(act_event, &act_event_txt);
  txt_out_stream_ << act_event_txt;
  if (FLAGS_force_flush_act_event) { txt_out_stream_.Flush(); }
}

ActEventLogger::ActEventLogger()
    : bin_out_stream_(LocalFS(), JoinPath(LogDir(), act_event_bin_filename_)),
      txt_out_stream_(LocalFS(), JoinPath(LogDir(), act_event_txt_filename_)) {}

}  // namespace oneflow
