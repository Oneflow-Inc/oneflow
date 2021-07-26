#include "oneflow/core/common/data_type.h"

// This header file must be included instead of
// forward declaring PyArrayObject, or compile error
// will occur
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace oneflow {

namespace numpy {

Maybe<int> OFDataTypeToNumpyType(DataType of_data_type);

Maybe<DataType> NumpyTypeToOFDataType(int np_array_type);

Maybe<DataType> GetOFDataTypeFromNpArray(PyArrayObject* array);

}  // namespace numpy
}  // namespace oneflow
