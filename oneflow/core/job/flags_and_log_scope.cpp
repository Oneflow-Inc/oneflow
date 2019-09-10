#include "oneflow/core/job/flags_and_log_scope.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/file_system.h"

DECLARE_bool(grpc_use_no_signal);

namespace oneflow {

namespace {

#define OF_VERSION_MAJOR "0"
#define OF_VERSION_MINOR "1"
#define OF_VERSION_PATCH "0"
#define OF_VERSION OF_VERSION_MAJOR "." OF_VERSION_MINOR "." OF_VERSION_PATCH

std::string BuildVersionString() {
  static const HashMap<std::string, std::string> month_word2num = {
      {"Jan", "01"}, {"Feb", "02"}, {"Mar", "03"}, {"Apr", "04"}, {"May", "05"}, {"Jun", "06"},
      {"Jul", "07"}, {"Aug", "08"}, {"Sep", "09"}, {"Oct", "10"}, {"Nov", "11"}, {"Dec", "12"},
  };
  static const std::string date_str(__DATE__);
  std::string day = date_str.substr(4, 2);
  StringReplace(&day, ' ', '0');
  return OF_VERSION " (" + date_str.substr(7) + month_word2num.at(date_str.substr(0, 3)) + day + "."
         + __TIME__ + ")";
}

std::string LogDir(const std::string& log_dir) {
  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);
  std::string v = log_dir + "/" + std::string(hostname);
  return v;
}
}  // namespace

FlagsAndLogScope::FlagsAndLogScope(const ConfigProto& config, const char* binary_name) {
  FLAGS_log_dir = LogDir(config.cpp_flags_conf().log_dir());
  FLAGS_logtostderr = config.cpp_flags_conf().logtostderr();
  FLAGS_logbuflevel = config.cpp_flags_conf().logbuflevel();
  FLAGS_grpc_use_no_signal = config.cpp_flags_conf().grpc_use_no_signal();
  google::InitGoogleLogging(binary_name);
  gflags::SetVersionString(BuildVersionString());
  LocalFS()->RecursivelyCreateDirIfNotExist(FLAGS_log_dir);
}

FlagsAndLogScope::~FlagsAndLogScope() {}

}  // namespace oneflow
