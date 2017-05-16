/*
 * grpc_call.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRPC_CALL_H
#define GRPC_CALL_H

#include "grpc++/grpc++.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/server_builder.h"

namespace oneflow {
 
template <class Service>
class UntypedCall {
  public:
    virtual ~UntypedCall() {}

    virtual RequestReceived(Service* service) = 0;

    class Tag {
      public:
        enum Callback {kRequestReceived};
        Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {}

        void OnCompleted(Service* service) {
          switch(callback_) {
            case kRequestReceived:
              call_->RequestReceived(service);
          }
        }
      private:
        UntypedCall* const call_;
        Callback callback_;
    };
}; 
 
}


#endif /* !GRPC_CALL_H */
