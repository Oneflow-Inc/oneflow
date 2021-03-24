import os
import oneflow
local_label=""
if os.getenv("ONEFLOW_RELEASE_VERSION"):
    version=os.getenv("ONEFLOW_RELEASE_VERSION")
else:
