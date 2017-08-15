#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

std::string StatusToString(Status s) {
  std::string result;
  switch (s) {
    case Status::OK: result = "OK"; break;
    case Status::FAILED_PRECONDITION: result = "Failed precondition"; break;
    case Status::NOT_FOUND: result = "Not found"; break;
    case Status::PERMISSION_DENIED: result = "Permission denied"; break;
    case Status::UNIMPLEMENTED: result = "Unimplemented"; break;
    default: result = "Unknown code " + std::to_string(s); break;
  }
  return result;
}

}  // namespace oneflow
