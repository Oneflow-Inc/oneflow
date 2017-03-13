#ifndef ONEFLOW_SERVER_LIB_H_
#define ONEFLOW_SERVER_LIB_H_

#include <iostream>
#include <memory>

//#include "macros.h"
#include "oneflow_server.pb.h"

#define OF_DISALLOW_COPY_AND_ASSIGN(TypeName) \  
  TypeName(const TypeName&) = delete;         \  
  void operator=(const TypeName&) = delete  

namespace oneflow {

// This library supports a registration/factory-based mechanism for
// creating TensorFlow server objects. Each server implementation must
// have an accompanying implementation of ServerFactory, and create a
// static "registrar" object that calls `ServerFactory::Register()`
// with an instance of the factory class. See "rpc/grpc_server_lib.cc"
// for an example.

// Represents a single TensorFlow server that exports Master and Worker
// services.
class ServerInterface {
 public:
  ServerInterface() {}
  virtual ~ServerInterface() {}

  // Starts the server running asynchronously. Returns OK on success, otherwise
  // returns an error.
  virtual int Start() = 0;

  // Stops the server asynchronously. Returns OK on success, otherwise returns
  // an error.
  //
  // After calling `Stop()`, the caller may call `Join()` to block until the
  // server has stopped.
  virtual int Stop() = 0;

  // Blocks until the server has stopped. Returns OK on success, otherwise
  // returns an error.
  virtual void Join() = 0;

  // Returns a target string that can be used to connect to this server using
  // `tensorflow::NewSession()`.
  virtual const std::string target() const = 0;

 private:
  OF_DISALLOW_COPY_AND_ASSIGN(ServerInterface);
};

class ServerFactory {
 public:
  // Creates a new server based on the given `server_def`, and stores
  // it in `*out_server`. Returns OK on success, otherwise returns an
  // error.
  virtual int NewServer(const ServerDef& server_def,
                           std::unique_ptr<ServerInterface>* out_server) = 0;

  // Returns true if and only if this factory can create a server
  // based on the given `server_def`.
  virtual bool AcceptsOptions(const ServerDef& server_def) = 0;

  virtual ~ServerFactory() {}

  // For each `ServerFactory` subclass, an instance of that class must
  // be registered by calling this method.
  //
  // The `server_type` must be unique to the server factory.
  static void Register(const std::string& server_type, ServerFactory* factory);

  // Looks up a factory that can create a server based on the given
  // `server_def`, and stores it in `*out_factory`. Returns OK on
  // success, otherwise returns an error.
  static int GetFactory(const ServerDef& server_def,
                           ServerFactory** out_factory);
};

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
int NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server);

}  // namespace tensorflow

#endif  // ONEFLOW_SERVER_LIB_H_
