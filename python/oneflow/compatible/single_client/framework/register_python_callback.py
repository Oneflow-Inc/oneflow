import oneflow._oneflow_internal
from oneflow.compatible.single_client.python.eager import (
    interpreter_callback as interpreter_callback,
)
from oneflow.compatible.single_client.python.framework import (
    python_callback as python_callback,
)

python_callback.interpreter_callback = interpreter_callback
