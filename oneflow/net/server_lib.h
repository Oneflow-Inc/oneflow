#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_

#include <memory>

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
  virtual void Start() = 0;

  // Stops the server asynchronously. Returns OK on success, otherwise returns
  // an error.
  //

  // Blocks until the server has stopped. Returns OK on success, otherwise
  // returns an error.
  virtual void Join() = 0;

  // Returns a target string that can be used to connect to this server using
  // `tensorflow::NewSession()`.
  //virtual const std::string target() const = 0;

 private:
};

class ServerFactory {
 public:
  // Creates a new server based on the given `server_def`, and stores
  // it in `*out_server`. Returns OK on success, otherwise returns an
  // error.
  virtual void NewServer(std::unique_ptr<ServerInterface>* out_server) = 0;

  // Returns true if and only if this factory can create a server
  // based on the given `server_def`.
  virtual bool AcceptsOptions() = 0;

  virtual ~ServerFactory() {}

  // For each `ServerFactory` subclass, an instance of that class must
  // be registered by calling this method.
  //
  // The `server_type` must be unique to the server factory.
  static void Register(const std::string& server_type, ServerFactory* factory);

  // Looks up a factory that can create a server based on the given
  // `server_def`, and stores it in `*out_factory`. Returns OK on
  // success, otherwise returns an error.
  static void GetFactory(ServerFactory** out_factory);
};

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
void NewServer(std::unique_ptr<ServerInterface>* out_server);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_
