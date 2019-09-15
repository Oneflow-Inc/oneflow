from __future__ import absolute_import

import threading
import oneflow.python.framework.inter_user_job_util as inter_user_job_util
import oneflow.python.framework.remote_blob as remote_blob_util

class OutRemoteBlobsStatus(object):
    def __init__(self):
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
        return self._GetResultNdarray(self.out_remote_blob_pullers_)

    # user api
    def async_get(self, callback):
        assert self.inited_
        assert self.data_delivered_ == False
        pullers_cnt = self._GetPullersCnt()
        def Callback():
            assert self.finished_cnt_ <= pullers_cnt
            if self.finished_cnt_ == pullers_cnt:
                callback(self._GetResultNdarray(self.out_remote_blob_pullers_))
        try: 
            self.cond_var_.acquire()
            if self.finished_cnt_ == pullers_cnt:
                Callback()
            else:
                self.async_get_callback_ = Callback
        finally: self.cond_var_.release()
        self.data_delivered_ = True

    def AddResult(self, out_remote_blobs):
        assert self.inited_ == False
        pullers = self._MakeRemoteBlobPullers(out_remote_blobs)
        self.out_remote_blob_pullers_.append(pullers)
        for puller in self._FlatRemoteBlobPullers(pullers):
            puller.AsyncPull(self._FinishCallback)
        return self

    def SetResult(self, out_remote_blobs):
        assert self.inited_ == False
        assert isinstance(self.out_remote_blob_pullers_, list) 
        assert len(self.out_remote_blob_pullers_) == 0 
        pullers = self._MakeRemoteBlobPullers(out_remote_blobs)
        self.out_remote_blob_pullers_ = pullers
        for puller in self._FlatRemoteBlobPullers(pullers):
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

    def _GetResultNdarray(self, pullers):
        assert self.inited_
        if isinstance(pullers, _RemoteBlobPuller):
            return pullers.result_ndarray
        if isinstance(pullers, list) or isinstance(pullers, tuple):
            ret = [self._GetResultNdarray(x) for x in pullers]
            if isinstance(pullers, tuple): ret = tuple(ret)
            return ret
        if isinstance(pullers, dict):
            return {
                k : self._GetResultNdarray(v) for k, v in pullers.items()
            }
        raise NotImplementedError
    
    def _GetPullersCnt(self):
       cnt = 0
       for _ in self._FlatRemoteBlobPullers(self.out_remote_blob_pullers_): cnt += 1
       return cnt

    def _FlatRemoteBlobPullers(self, pullers):
        if isinstance(pullers, _RemoteBlobPuller):
            yield pullers
        elif isinstance(pullers, list) or isinstance(pullers, tuple):
            for elem in pullers:
                for x in self._FlatRemoteBlobPullers(elem): yield x
        elif isinstance(pullers, dict):
            for _, v in pullers.items():
                for x in self._FlatRemoteBlobPullers(v): yield x
        else:
            raise NotImplementedError

    def _MakeRemoteBlobPullers(self, out_remote_blobs):
        if isinstance(out_remote_blobs, remote_blob_util.RemoteBlob):
            return _RemoteBlobPuller(out_remote_blobs, self.cond_var_)
        if isinstance(out_remote_blobs, list) or isinstance(out_remote_blobs, tuple):
            ret = [self._MakeRemoteBlobPullers(x) for x in out_remote_blobs]
            if isinstance(out_remote_blobs, tuple): ret = tuple(ret)
            return ret
        if isinstance(out_remote_blobs, dict):
            return {
                k : self._MakeRemoteBlobPullers(v) for k, v in out_remote_blobs.items()
            }
        raise NotImplementedError

class _RemoteBlobPuller(object):
    def __init__(self, remote_blob, cond_var):
        self.op_name_ = remote_blob.op_name
        self.result_ndarray_ = None

    @property
    def result_ndarray(self):
        assert self.result_ndarray_ is not None
        return self.result_ndarray_

    @property
    def is_finished(self):
        return self.result_ndarray_ is not None

    def AsyncPull(self, pull_cb):
        def PullCallback(of_blob):
            self.result_ndarray_ = of_blob.CopyToNdarray()
            pull_cb()
        inter_user_job_util.AsyncPull(self.op_name_, PullCallback)
