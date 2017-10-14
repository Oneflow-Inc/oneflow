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

  void AddFd(int fd, std::function<void()> read_handler,
             std::function<void()> write_handler);
  void AddFdWithOnlyReadHandler(int fd, std::function<void()> read_handler);

  void Start();
  void Stop();

 private:
  struct IOHandler {
    IOHandler() {
      read_handler = []() { UNEXPECTED_RUN(); };
      write_handler = []() { UNEXPECTED_RUN(); };
      fd = -1;
    }
    std::function<void()> read_handler;
    std::function<void()> write_handler;
    int fd;
  };

  void AddFd(int fd, std::function<void()>* read_handler,
             std::function<void()>* write_handler);

  void EpollLoop();
  static const int max_event_num_;

  int epfd_;
  epoll_event* ep_events_;
  std::forward_list<IOHandler*> io_handlers_;
  int break_epoll_loop_fd_;
  std::thread thread_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_IO_EVENT_POLLER_H_
