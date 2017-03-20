#ifndef PLATFORM_ENV_H_
#define PLATFORM_ENV_H_

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
                      std::function<void()> fn) = 0;
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
