from oneflow.compatible.single_client.python.framework import (
    python_callback as python_callback,
)
from oneflow.compatible.single_client.python.eager import (
    interpreter_callback as interpreter_callback,
)
import oneflow._oneflow_internal

python_callback.interpreter_callback = interpreter_callback
