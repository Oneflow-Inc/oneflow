#include <iostream>
#include "platform_env.h"
#include <thread>
#include <string>

namespace oneflow {

namespace {
 
class StdThread : public Thread {
 public:
  StdThread(const ThreadOptions& thread_options, const std::string& name,
	    std::function<void()> fn) : thread_(fn) {}
  ~StdThread() {thread_.join();}
 
 private:
  std::thread thread_;
};

class PosixEnv : public Env {
 public:
  PosixEnv() {}
  ~PosixEnv() {}

  Thread* StartThread(const ThreadOptions& thread_options, const std::string& name, std::function<void()> fn) override {
    std::cout<<"platform posix env.h ------"<<std::endl;
    return new StdThread(thread_options, name, fn);
  }
};

}

}
