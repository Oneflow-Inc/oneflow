#ifndef _PROTO_PROTO_IO_H_
#define _PROTO_PROTO_IO_H_

#include <io.h>
#include <string>
#include <glog/logging.h>

#include "google/protobuf/message.h"
#include "google/protobuf/descriptor.h"

namespace caffe {

using ::google::protobuf::Message;

bool ParseProtoFromString(const std::string& str, Message* proto);
bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline void ParseProtoFromStringOrDie(const std::string& str, Message* proto) {
  CHECK(ParseProtoFromString(str, proto));
}
void PrintProtoToString(const Message& proto, std::string* str);
inline bool ReadProtoFromTextFile(const std::string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const std::string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const std::string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const std::string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const std::string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const std::string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}
}  // namespace caffe

#endif   // _PROTO_PROTO_IO_H_
