#include <mutex>

#include "memory/blob.h"
#include "caffe.pb.h"

namespace caffe {
namespace io {

class Stream {
public:
  Stream() {}
  virtual ~Stream() {}

  virtual void ReadOneStep() = 0;
  virtual void Seek(size_t pos) = 0;
  virtual size_t Tell() = 0;
  virtual std::string key() = 0;
  virtual std::string value() = 0;
  virtual bool Valid() = 0;

  virtual void Put(const std::string& key, const std::string& value) = 0;
  virtual void Done() = 0;
  virtual bool AtEnd() const = 0;

  static Stream *CreateForRead(const std::string &path,
    bool allow_null = false);

private:
  Stream(const Stream& other) = delete;
  Stream& operator=(const Stream& other) = delete;
};

class FileSystem {
public:

  static FileSystem *GetInstance(const std::string &path);
  virtual ~FileSystem() {}

  virtual Stream *Open(const std::string &path,
    const char* const flag,
    bool allow_null = false) = 0;

  virtual Stream *OpenForRead(const std::string &path,
    bool allow_null = false) = 0;
  virtual Stream *OpenForWrite(const std::string &path,
    bool allow_null = true) = 0;
};

class LocalFileSystem : public FileSystem {
public:
  virtual ~LocalFileSystem() {}
  
  // may support more LocalFileSystems
  virtual Stream *Open(const std::string &path,
    const char* const flag,
    bool allow_null);

  virtual Stream *OpenForRead(const std::string &path, bool allow_null);
  virtual Stream *OpenForWrite(const std::string &path, bool allow_null);

  inline static LocalFileSystem *GetInstance(void) {
    static LocalFileSystem instance;
    return &instance;
  }

private:
  LocalFileSystem() {}
};

}
}