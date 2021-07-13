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
from oneflow.python.utils.data.datapipes.iter.listdirfiles import (
    ListDirFilesIterDataPipe as ListDirFiles,
)
from oneflow.python.utils.data.datapipes.iter.loadfilesfromdisk import (
    LoadFilesFromDiskIterDataPipe as LoadFilesFromDisk,
)
from oneflow.python.utils.data.datapipes.iter.readfilesfromtar import (
    ReadFilesFromTarIterDataPipe as ReadFilesFromTar,
)
from oneflow.python.utils.data.datapipes.iter.readfilesfromzip import (
    ReadFilesFromZipIterDataPipe as ReadFilesFromZip,
)
from oneflow.python.utils.data.datapipes.iter.routeddecoder import (
    RoutedDecoderIterDataPipe as RoutedDecoder,
)

# Functional DataPipe
from oneflow.python.utils.data.datapipes.iter.callable import (
    MapIterDataPipe as Map,
    CollateIterDataPipe as Collate,
    TransformsIterDataPipe as Transforms,
)
from oneflow.python.utils.data.datapipes.iter.combining import (
    ConcatIterDataPipe as Concat,
    ZipIterDataPipe as Zip,
)
from oneflow.python.utils.data.datapipes.iter.combinatorics import (
    SamplerIterDataPipe as Sampler,
    ShuffleIterDataPipe as Shuffle,
)
from oneflow.python.utils.data.datapipes.iter.grouping import (
    BatchIterDataPipe as Batch,
    BucketBatchIterDataPipe as BucketBatch,
    GroupByKeyIterDataPipe as GroupByKey,
)
from oneflow.python.utils.data.datapipes.iter.selecting import (
    FilterIterDataPipe as Filter,
)


__all__ = [
    "ListDirFiles",
    "LoadFilesFromDisk",
    "ReadFilesFromTar",
    "ReadFilesFromZip",
    "RoutedDecoder",
    "GroupByKey",
    "Batch",
    "BucketBatch",
    "Collate",
    "Concat",
    "Filter",
    "Map",
    "Sampler",
    "Shuffle",
    "Transforms",
    "Zip",
]
