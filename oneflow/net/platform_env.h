#ifndef PLATFORM_ENV_H_
#define PLATFORM_ENV_H_
#include <functional>

namespace oneflow {

class Thread;
struct ThreadOptions;

class Env {
 public:
  Env();
  virtual ~Env() = default;
  static Env* Default();

  virtual Thread* StartThread(const ThreadOptions& thread_options, 
			      const std::string& name, 
			      std::function<void()> fn) = 0;
};

class EnvWrapper : public Env {
 public:
  Thread* StartThread(const ThreadOptions& thread_options,
                      const std::string& name,
                      std::function<void()> fn) override {
    return target_->StartThread(thread_options, name, fn);
  }
 private:
  Env* target_;
};

class Thread {
 public:
  Thread() {}
  virtual ~Thread();

 private:
  Thread(const Thread&);
  void operator=(const Thread&); 
};

struct ThreadOptions {
  size_t stack_size = 0;
  size_t guard_size = 0;
};

}

#endif
