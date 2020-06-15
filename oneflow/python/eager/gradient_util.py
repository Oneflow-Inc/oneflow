from __future__ import absolute_import

import oneflow.python.framework.session_context as session_ctx


def SetBwUsedBlobObject4UniqueName(blob_unique_name, blob_object):
    name2blob_object = (
        session_ctx.GetDefaultSession().eager_unique_name2bw_used_blob_object
    )
    assert blob_unique_name not in name2blob_object
    name2blob_object[blob_unique_name] = blob_object


def HasBwUsedBlobObject4UniqueName(blob_unique_name):
    name2blob_object = (
        session_ctx.GetDefaultSession().eager_unique_name2bw_used_blob_object
    )
    return blob_unique_name in name2blob_object


def GetBwUsedBlobObject4UniqueName(blob_unique_name):
    name2blob_object = (
        session_ctx.GetDefaultSession().eager_unique_name2bw_used_blob_object
    )
    assert blob_unique_name in name2blob_object
    return name2blob_object[blob_unique_name]


def ClearBwUsedBlobObject4UniqueName(blob_unique_name):
    name2blob_object = (
        session_ctx.GetDefaultSession().eager_unique_name2bw_used_blob_object
    )
    assert blob_unique_name in name2blob_object
    del name2blob_object[blob_unique_name]
