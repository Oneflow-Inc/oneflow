#ifndef PLATFORM_ENV_H_
#define PLATFORM_ENV_H_

#include <functional>
#include <thread>

namespace oneflow {

class Thread;
struct ThreadOptions;

class Env {
 public:
  Env();
  virtual ~Env() = default;
  static Env* Default();
};

class Thread {
 public:
  Thread() {}
  //~Thread();

 private:
  //Thread(const Thread&);
  //void operator=(const Thread&); 
};

struct ThreadOptions {
  size_t stack_size = 0;
  size_t guard_size = 0;
};

Thread* StartThread(const ThreadOptions& thread_options,
                              const std::string& name,
                              std::function<void()> fn);

}

#endif
