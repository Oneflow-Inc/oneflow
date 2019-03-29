#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_IO_EVENT_POLLER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_IO_EVENT_POLLER_H_

#include "oneflow/core/comm_network/epoll/socket_message.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class IOEventPoller final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IOEventPoller);
  IOEventPoller();
  ~IOEventPoller();

  void AddFd(int32_t fd, std::function<void()> read_handler, std::function<void()> write_handler);
  void AddFdWithOnlyReadHandler(int32_t fd, std::function<void()> read_handler);

  void Start();
  void Stop();

 private:
  struct IOHandler {
    IOHandler() {
      read_handler = []() { UNIMPLEMENTED(); };
      write_handler = []() { UNIMPLEMENTED(); };
      fd = -1;
    }
    std::function<void()> read_handler;
    std::function<void()> write_handler;
    int32_t fd;
  };

  void AddFd(int32_t fd, std::function<void()>* read_handler, std::function<void()>* write_handler);

  void EpollLoop();
  static const int32_t max_event_num_;

  int32_t epfd_;
  epoll_event* ep_events_;
  std::forward_list<IOHandler*> io_handlers_;
  int32_t break_epoll_loop_fd_;
  std::thread thread_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_IO_EVENT_POLLER_H_
