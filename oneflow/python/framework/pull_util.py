from __future__ import absolute_import

import threading
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.local_blob as local_blob_util

class FutureRemoteBlobs(object):
    def __init__(self, session):
        self.session_ = session
        self.cond_var_ = threading.Condition()
        self.out_remote_blob_pullers_ = []
        self.inited_ = False
        self.finished_cnt_ = 0
        self.data_delivered_ = False
        self.async_get_callback_ = lambda : None

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
                callback(self._TrySyncAndGetResultNdarray(self.out_remote_blob_pullers_))
        try: 
            self.cond_var_.acquire()
            if self.finished_cnt_ == pullers_cnt:
                Callback()
            else:
                self.async_get_callback_ = Callback
        finally: self.cond_var_.release()
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
    
    def Inited(self):
        assert self.inited_ == False
        self.inited_ = True
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
        while self.finished_cnt_ != pullers_cnt: self.cond_var_.wait()
        self.cond_var_.release()

    def _TrySyncAndGetResultNdarray(self, pullers):
        if self.session_.HasAnyCallbackAfterFunctionReturn(): self.session_.Sync()
        return self._GetResultNdarray(pullers)

    def _GetResultNdarray(self, pullers):
        assert self.inited_
        if isinstance(pullers, _BlobPuller):
            return pullers.result
        if isinstance(pullers, (list, tuple)):
            return type(pullers)(self._GetResultNdarray(x) for x in pullers)
        if isinstance(pullers, dict):
            return {
                k : self._GetResultNdarray(v) for k, v in pullers.items()
            }
        raise NotImplementedError
    
    def _GetPullersCnt(self):
       cnt = 0
       for _ in self._FlatConsistentBlobPullers(self.out_remote_blob_pullers_): cnt += 1
       return cnt

    def _FlatConsistentBlobPullers(self, pullers):
        if isinstance(pullers, _BlobPuller):
            for x in pullers.FlatConsistentBlobPullers(): yield x
        elif isinstance(pullers, list) or isinstance(pullers, tuple):
            for elem in pullers:
                for x in self._FlatConsistentBlobPullers(elem): yield x
        elif isinstance(pullers, dict):
            for _, v in pullers.items():
                for x in self._FlatConsistentBlobPullers(v): yield x
        else:
            raise NotImplementedError

    def _MakeRemoteBlobPullers(self, out_remote_blobs):
        if isinstance(out_remote_blobs, remote_blob_util.ConsistentBlob):
            return _ConsistentBlobPuller(out_remote_blobs, self.session_)
        if isinstance(out_remote_blobs, remote_blob_util.MirrorBlob):
            return _MirrorBlobPuller(out_remote_blobs, self.session_)
        if isinstance(out_remote_blobs, list) or isinstance(out_remote_blobs, tuple):
            return type(out_remote_blobs)(self._MakeRemoteBlobPullers(x) for x in out_remote_blobs)
        if isinstance(out_remote_blobs, dict):
            return {
                k : self._MakeRemoteBlobPullers(v) for k, v in out_remote_blobs.items()
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
    def __init__(self, remote_blob, session):
        _BlobPuller.__init__(self, session)
        self.result_ = None
        self.op_name_ = remote_blob.op_name
        
    @property
    def result(self):
        assert self.result_ is not None
        return self.result_

    def FlatConsistentBlobPullers(self): yield self

    def AsyncPull(self, pull_cb):
        def PullCallback(of_blob):
            self.result_ = of_blob.CopyToBlob()
            pull_cb()
        self.session_.AsyncPull(self.op_name_, PullCallback)

class _MirrorBlobPuller(_BlobPuller):
    def __init__(self, mirror_blob_def, session):
        _BlobPuller.__init__(self, session)
        self.sub_pullers_ = tuple(_ConsistentBlobPuller(x, self.session_)
                                 for x in mirror_blob_def.sub_consistent_blob_list)
        self.local_mirror_blob_ = None
        
    @property
    def result(self):
        if self.local_mirror_blob_ is not None: return self.local_mirror_blob_
        ndarray_list = [x.result.ndarray() for x in self.sub_pullers_]
        self.local_mirror_blob_ = local_blob_util.LocalMirrorBlob(ndarray_list)
        return self.local_mirror_blob_

    def FlatConsistentBlobPullers(self):
        for x in self.sub_pullers_: yield x
