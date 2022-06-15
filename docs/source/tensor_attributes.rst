.. currentmodule:: oneflow

Tensor Attributes
=============================================================
Each local ``oneflow.Tensor`` has a :class:`oneflow.dtype`, :class:`oneflow.device`, and global ``oneflow.Tensor`` has a :class:`oneflow.dtype`, :class:`oneflow.placement`, :class:`oneflow.sbp`.


.. autosummary::
    :toctree: generated
    :nosignatures:

    device
    placement
    env.all_device_placement
    sbp.sbp