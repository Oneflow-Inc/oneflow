/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_IO_EVENT_POLLER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_IO_EVENT_POLLER_H_

#include "oneflow/core/comm_network/epoll/socket_message.h"

#ifdef OF_PLATFORM_POSIX

namespace oneflow {

class IOEventPoller final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IOEventPoller);
  IOEventPoller();
  ~IOEventPoller();

  void AddFd(int fd, std::function<void()> read_handler, std::function<void()> write_handler);
  void AddFdWithOnlyReadHandler(int fd, std::function<void()> read_handler);

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
    int fd;
  };

  void AddFd(int fd, std::function<void()>* read_handler, std::function<void()>* write_handler);

  void EpollLoop();
  static const int max_event_num_;

  int epfd_;
  epoll_event* ep_events_;
  std::forward_list<IOHandler*> io_handlers_;
  int break_epoll_loop_fd_;
  std::thread thread_;
};

}  // namespace oneflow

#endif  // OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_IO_EVENT_POLLER_H_
