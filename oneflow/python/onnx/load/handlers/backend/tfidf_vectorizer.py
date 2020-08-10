"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import tensorflow as tf

from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.load.handlers.handler import onnx_op


@onnx_op("TfIdfVectorizer")
class TfIdfVectorizer(BackendHandler):
    @classmethod
    def args_check(cls, node, **kwargs):
        if "pool_int64s" in node.attrs and "pool_strings" in node.attrs:
            raise ValueError(
                "Cannot set the pool_int64s and pool_strings in an input at the same time."
            )

    @classmethod
    def _prepare_ngrams(cls, x, n, skip):
        # This method transform input into n-grams for a specific skip
        # input x is a 1D tensor
        # ex1: x=[1,2,3,4] n=1 skip=0 output=[[1],[2],[3],[4]]
        # ex2: x=[1,2,3,4] n=2 skip=0 output=[[1,2],[2,3],[3,4]]
        # ex3: x=[1,2,3,4] n=2 skip=1 output=[[1,3],[2,4]]
        count = x.shape[0] - n + 1 - skip
        multiplier = skip + 1
        ngrams = [x[i * multiplier : i * multiplier + count] for i in range(n)]
        ngrams = tf.stack(ngrams)
        ngrams = tf.transpose(ngrams, [1, 0])
        return ngrams

    @classmethod
    def _calc_ngram_skip(cls, x, pool, n, skip=0):
        # This method calculates ngram counts for specific n and skip

        # Make pool into an array of ngrams
        pool = np.reshape(pool, (int(len(pool) / n), n))

        # Make input as an array of ngrams
        new_x = cls._prepare_ngrams(x, n, skip)

        # Loop through the ngram targets in the pool
        tensor_list = []
        for i in range(len(pool)):
            ngram_count = tf.map_fn(
                lambda in_x: tf.where(
                    tf.reduce_all(
                        tf.equal(in_x, tf.constant(pool[i], dtype=new_x.dtype))
                    ),
                    tf.constant([1]),
                    tf.constant([0]),
                ),
                new_x,
                dtype=tf.int32,
            )
            ngram_count = tf.math.count_nonzero(ngram_count, dtype=tf.int32)
            ngram_count = tf.reshape(ngram_count, [1])
            tensor_list.append(ngram_count)

        return tf.concat(tensor_list, 0)

    @classmethod
    def _calc_ngram(cls, x, pool, n, max_skip):
        # This method calculates ngram counts for a specific n and
        # all allowable skips

        # For 1gram, skip is not in use. Not clearly described in ONNX
        # spec, this code logic is based on observation of ONNX examples,
        # tf_batch_uniandbigrams_skip5 and tf_uniandbigrams_skip5,
        # where the 1-gram results [0, 3, 0, 0] and [0, 3, 1, 0]
        # are not the accumulated counts from multiple skips.
        if n == 1:
            return cls._calc_ngram_skip(x, pool, n)

        # Loop through maximum allowable skip count and sum up the results
        result = tf.zeros([int(len(pool) / n)], dtype=tf.int32)
        max_allowable_skip = np.minimum(
            max_skip, int((int(x.shape[0]) - 1) / (n - 1) - 1)
        )

        for skip in range(max_allowable_skip + 1):
            # For each skip calculate the ngram counts
            result += cls._calc_ngram_skip(x, pool, n, skip)

        return result

    @classmethod
    def version_9(cls, node, **kwargs):
        input_tensor = kwargs["tensor_dict"][node.inputs[0]]
        mode = node.attrs.get("mode")
        max_skip_count = node.attrs.get("max_skip_count")
        min_gram_len = node.attrs.get("min_gram_length")
        max_gram_len = node.attrs.get("max_gram_length")
        ngram_counts = node.attrs.get("ngram_counts")
        ngram_indexes = node.attrs.get("ngram_indexes")
        pool_int64s = node.attrs.get("pool_int64s")
        pool_strings = node.attrs.get("pool_strings")
        weights = node.attrs.get("weights", np.ones(len(ngram_indexes)))

        def process_ngram(input_t):
            # This is the main method that processes and produces ngram counts
            # for one row of inputs regardless of the operator input dimension.
            size = len(ngram_indexes)
            new_ngram_counts = np.append(ngram_counts, size)
            result_ngram = np.zeros(size)
            for i in range(len(new_ngram_counts) - 1):
                gram_len = i + 1
                count = new_ngram_counts[i + 1] - new_ngram_counts[i]
                total_len = count * gram_len
                if gram_len >= min_gram_len and gram_len <= max_gram_len:
                    idx = ngram_indexes[new_ngram_counts[i] : new_ngram_counts[i + 1]]
                    process_pool = (
                        pool_int64s[
                            new_ngram_counts[i] : new_ngram_counts[i] + total_len
                        ]
                        if pool_int64s is not None
                        else pool_strings[
                            new_ngram_counts[i] : new_ngram_counts[i] + total_len
                        ]
                    )
                    result = cls._calc_ngram(
                        input_t, process_pool, gram_len, max_skip_count
                    )
                    idx = tf.constant(idx, shape=[len(idx), 1])
                    result_ngram = result_ngram + tf.scatter_nd(idx, result, [size])
            return result_ngram

        # The input can be either 1d or 2d. Need to loop through
        # each element for 2d inputs
        n = len(input_tensor.shape)
        final_out = (
            [process_ngram(input_tensor[i]) for i in range(input_tensor.shape[0])]
            if n > 1
            else process_ngram(input_tensor)
        )
        tf_out = tf.cast(final_out, tf.float32)

        # Apply the mode based of the TF output
        if mode == "IDF":
            return [tf.minimum(tf_out, 1) * weights]
        elif mode == "TFIDF":
            return [tf_out * weights]
        else:
            return [tf_out]
