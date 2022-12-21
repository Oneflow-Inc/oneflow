# Monkey patch to not ship libjvm.so in pypi wheels
import sys

from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

nv_libs = [
    "libcudnn_adv_infer.so.8"
    "libcudnn_adv_train.so.8"
    "libcudnn_cnn_infer.so.8"
    "libcudnn_cnn_train.so.8"
    "libcudnn_ops_infer.so.8"
    "libcudnn_ops_train.so.8"
    "libcudnn.so.8"
    "libcublas.so.11"
]
for p in POLICIES:
    for nv_lib in nv_libs:
        p["lib_whitelist"].append("libjvm.so")

if __name__ == "__main__":
    sys.exit(main())
