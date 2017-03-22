#include "platform_env.h"

namespace oneflow {
  class StdThread;
  class ThreadOptions;
 
  class StdThread : public Thread {
   public:
    StdThread(const ThreadOptions& thread_options, const std::string& name,
            std::function<void()> fn) : thread_(fn) {}
    ~StdThread() {thread_.join();}

   private:
    std::thread thread_;
};  

  Thread* StartThread(const ThreadOptions& thread_options,
			   const std::string& name,
			   std::function<void()> fn) {

    return new StdThread(thread_options, name, fn);    
  }


}
