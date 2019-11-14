oneflow
===================================
.. currentmodule:: oneflow
.. automodule:: oneflow
    :members:

Operators
----------------------------------

.. autofunction:: get_variable
.. autofunction:: gather
.. autofunction:: identity
.. autofunction:: matmul
.. autofunction:: parallel_cast
.. autofunction:: reshape
.. autofunction:: slice
.. autofunction:: transpose
.. autofunction:: truncated_normal
.. autofunction:: unsorted_batch_segment_sum
.. autofunction:: unsorted_segment_sum
.. autofunction:: cast

Initializers
----------------------------------
.. autofunction:: ones_initializer
.. autofunction:: glorot_uniform_initializer
.. autofunction:: random_normal_initializer
.. autofunction:: random_uniform_initializer
.. autofunction:: truncated_normal_initializer
.. autofunction:: variance_scaling_initializer
.. autofunction:: xavier_uniform_initializer
.. autofunction:: zeros_initializer

System
----------------------------------

.. autofunction:: watch
.. autoclass:: input_blob_def
    :members:
    :undoc-members:
.. decorator:: function
.. autofunction:: function
.. autofunction:: get_default_job_set
.. autofunction:: reset_default_job_set
.. autofunction:: inter_job_reuse_mem_strategy
.. autofunction:: fixed_placement
.. autofunction:: device_prior_placement
.. autofunction:: get_default_job_set

Types
----------------------------------

.. autodata:: double
.. autodata:: float
.. autodata:: float32
.. autodata:: float64
.. autodata:: int32
.. autodata:: int64
.. autodata:: int8
