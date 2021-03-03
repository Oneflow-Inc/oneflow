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
from __future__ import absolute_import

import threading
import oneflow.python.framework.local_blob as local_blob_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api


class FutureRemoteBlobs(object):
    def __init__(self):
        self.inited_ = False

    def get(self):
        raise NotImplementedError

    def async_get(self, callback):
        raise NotImplementedError

    def SetResult(self, remote_blobs):
        raise NotImplementedError

    def Inited(self):
        assert self.inited_ is False
        self.inited_ = True
        return self


class LazyFutureRemoteBlobs(FutureRemoteBlobs):
    def __init__(self, session):
        super().__init__()
        self.session_ = session
        self.cond_var_ = threading.Condition()
        self.out_remote_blob_pullers_ = []
        self.finished_cnt_ = 0
        self.data_delivered_ = False
        self.async_get_callback_ = lambda: None

    # user api
    def get(self):
        assert self.inited_
        assert self.data_delivered_ == False
        self._Wait()
        self.data_delivered_ = True
        return self._TrySyncAndGetResultNdarray(self.out_remote_blob_pullers_)

    # user api
    def async_get(self, callback):
        assert self.inited_
        assert self.data_delivered_ == False
        pullers_cnt = self._GetPullersCnt()

        def Callback():
            assert self.finished_cnt_ <= pullers_cnt
            if self.finished_cnt_ == pullers_cnt:
                callback(
                    self._TrySyncAndGetResultNdarray(self.out_remote_blob_pullers_)
                )

        try:
            self.cond_var_.acquire()
            if self.finished_cnt_ == pullers_cnt:
                Callback()
            else:
                self.async_get_callback_ = Callback
        finally:
            self.cond_var_.release()
        self.data_delivered_ = True

    def SetResult(self, out_remote_blobs):
        assert self.inited_ == False
        assert isinstance(self.out_remote_blob_pullers_, list)
        assert len(self.out_remote_blob_pullers_) == 0
        pullers = self._MakeRemoteBlobPullers(out_remote_blobs)
        self.out_remote_blob_pullers_ = pullers
        for puller in self._FlatConsistentBlobPullers(pullers):
            puller.AsyncPull(self._FinishCallback)
        return self

    def _FinishCallback(self):
        self.cond_var_.acquire()
        self.finished_cnt_ += 1
        self.cond_var_.notify()
        self.async_get_callback_()
        self.cond_var_.release()

    def _Wait(self):
        pullers_cnt = self._GetPullersCnt()
        self.cond_var_.acquire()
        while self.finished_cnt_ != pullers_cnt:
            self.cond_var_.wait()
        self.cond_var_.release()

    def _TrySyncAndGetResultNdarray(self, pullers):
        if self.session_.HasAnyCallbackAfterFunctionReturn():
            self.session_.Sync()
        return self._GetResultLocalBlob(pullers)

    def _GetResultLocalBlob(self, pullers):
        assert self.inited_
        if isinstance(pullers, _BlobPuller):
            return pullers.result
        if isinstance(pullers, (list, tuple)):
            return type(pullers)(self._GetResultLocalBlob(x) for x in pullers)
        if isinstance(pullers, dict):
            return {k: self._GetResultLocalBlob(v) for k, v in pullers.items()}
        raise NotImplementedError

    def _GetPullersCnt(self):
        cnt = 0
        for _ in self._FlatConsistentBlobPullers(self.out_remote_blob_pullers_):
            cnt += 1
        return cnt

    def _FlatConsistentBlobPullers(self, pullers):
        if isinstance(pullers, _BlobPuller):
            for x in pullers.FlatConsistentBlobPullers():
                yield x
        elif isinstance(pullers, list) or isinstance(pullers, tuple):
            for elem in pullers:
                for x in self._FlatConsistentBlobPullers(elem):
                    yield x
        elif isinstance(pullers, dict):
            for _, v in pullers.items():
                for x in self._FlatConsistentBlobPullers(v):
                    yield x
        else:
            raise NotImplementedError

    def _MakeRemoteBlobPullers(self, out_remote_blobs):
        if isinstance(out_remote_blobs, oneflow_api.ConsistentBlob):
            return _ConsistentBlobPuller(out_remote_blobs, self.session_)
        if isinstance(out_remote_blobs, oneflow_api.MirroredBlob):
            return _MirroredBlobPuller(out_remote_blobs, self.session_)
        if isinstance(out_remote_blobs, list) or isinstance(out_remote_blobs, tuple):
            return type(out_remote_blobs)(
                self._MakeRemoteBlobPullers(x) for x in out_remote_blobs
            )
        if isinstance(out_remote_blobs, dict):
            return {
                k: self._MakeRemoteBlobPullers(v) for k, v in out_remote_blobs.items()
            }
        raise NotImplementedError


class _BlobPuller(object):
    def __init__(self, session):
        self.session_ = session

    def FlatConsistentBlobPullers(self):
        raise NotImplementedError

    @property
    def result(self):
        raise NotImplementedError


class _ConsistentBlobPuller(_BlobPuller):
    def __init__(self, consistent_blob, session):
        _BlobPuller.__init__(self, session)
        self.result_ = None
        self.consistent_blob_ = consistent_blob

    @property
    def result(self):
        assert self.result_ is not None
        return self.result_

    def FlatConsistentBlobPullers(self):
        yield self

    def AsyncPull(self, pull_cb):
        def PullCallback(of_blob):
            self.result_ = local_blob_util.MakeLocalBlob(
                [of_blob.CopyToNdarray()], self.consistent_blob_
            )
            pull_cb()

        self.session_.AsyncPull(self.consistent_blob_.op_name, PullCallback)


class _MirroredBlobPuller(_BlobPuller):
    def __init__(self, mirrored_blob, session):
        _BlobPuller.__init__(self, session)
        self.mirrored_blob_ = mirrored_blob
        self.sub_pullers_ = tuple(
            _ConsistentBlobPuller(x, self.session_)
            for x in mirrored_blob.sub_consistent_blob_list
        )
        self.local_mirrored_blob_ = None

    @property
    def result(self):
        if self.local_mirrored_blob_ is not None:
            return self.local_mirrored_blob_
        local_blob_list = [x.result for x in self.sub_pullers_]
        self.local_mirrored_blob_ = local_blob_util.MergeLocalBlobs(
            local_blob_list, self.mirrored_blob_
        )
        return self.local_mirrored_blob_

    def FlatConsistentBlobPullers(self):
        for x in self.sub_pullers_:
            yield x


class EagerFutureRemoteBlobs(FutureRemoteBlobs):
    def __init__(self):
        super().__init__()
        self.blob_getters_ = None

    def get(self):
        return self._GetResultLocalBlob(self.blob_getters_)

    def async_get(self, callback):
        assert callable(callback)
        callback(self._GetResultLocalBlob(self.blob_getters_))

    def SetResult(self, remote_blobs):
        assert self.inited_ is False
        assert self.blob_getters_ is None
        self.blob_getters_ = self._MakeRemoteBlobGetters(remote_blobs)
        return self

    def _MakeRemoteBlobGetters(self, remote_blobs):
        if isinstance(remote_blobs, (list, tuple)):
            return type(remote_blobs)(
                self._MakeRemoteBlobGetters(blob) for blob in remote_blobs
            )
        elif isinstance(remote_blobs, dict):
            return {k: self._MakeRemoteBlobGetters(v) for k, v in remote_blobs.items()}
        elif isinstance(remote_blobs, oneflow_api.EagerBlobTrait):
            return _EagerBlobGetter(remote_blobs)
        else:
            raise NotImplementedError

    def _GetResultLocalBlob(self, getter):
        assert self.inited_
        if isinstance(getter, _EagerBlobGetter):
            return getter.result
        elif isinstance(getter, (list, tuple)):
            return type(getter)(self._GetResultLocalBlob(g) for g in getter)
        elif isinstance(getter, dict):
            return {k: self._GetResultLocalBlob(v) for k, v in getter.items()}
        else:
            raise NotImplementedError(type(getter))


class _EagerBlobGetter(object):
    def __init__(self, eager_blob):
        assert isinstance(eager_blob, oneflow_api.EagerBlobTrait)
        self.eager_blob_ = eager_blob
        self.local_tensor_ = None

    @property
    def result(self):
        if self.local_tensor_ is not None:
            return self.local_tensor_

        self.local_tensor_ = local_blob_util.MakeLocalBlob4EagerBlob(self.eager_blob_)
        return self.local_tensor_
