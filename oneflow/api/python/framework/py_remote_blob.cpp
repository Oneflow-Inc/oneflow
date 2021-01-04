#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/py_remote_blob.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

class PyConsistentBlob: public ConsistentBlob, PyBlobDesc {
 public:
  using ConsistentBlob::ConsistentBlob;
};


}  // namespace compatible_py

}  // namespace oneflow
