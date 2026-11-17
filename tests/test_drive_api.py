"""Logic tests for gridnet.train.drive_api against a fake Drive v3 service (no network, no googleapiclient needed)."""
import os
import sys
import types

import pytest

from gridnet.train import drive_api


class HttpError(Exception):
    def __init__(self, status):
        self.resp = types.SimpleNamespace(status=status)


class MediaFileUpload:
    def __init__(self, path, mimetype, resumable, chunksize):
        self.path, self.mimetype, self.resumable, self.chunksize = path, mimetype, resumable, chunksize


class _Req:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _Upload:
    """The resumable request: next_chunk() yields progress twice, then the final metadata."""

    def __init__(self, store, file_id, name, parent, media):
        self.store, self.file_id, self.name, self.parent, self.media, self.calls = store, file_id, name, parent, media, 0

    def next_chunk(self):
        self.calls += 1
        if self.store.fail_first and self.calls == 1:
            raise HttpError(503)
        size = os.path.getsize(self.media.path)
        if self.calls < 3:
            return types.SimpleNamespace(resumable_progress=min(size, self.calls * self.media.chunksize)), None
        self.store.files[self.file_id] = {"id": self.file_id, "name": self.name, "size": str(size), "parent": self.parent}
        self.store.content[self.file_id] = open(self.media.path, "rb").read()
        return None, {"id": self.file_id, "name": self.name, "size": str(size)}


class FakeFiles:
    def __init__(self, store):
        self.store = store

    def list(self, q, fields, pageSize=100, pageToken=None, spaces=None):
        name = q.split("name = '")[1].split("'")[0] if "name = '" in q else None
        parent = q.split("'")[-2] if name is None else q.split(" and '")[1].split("'")[0]
        hits = [dict(f) for f in self.store.files.values()
                if f.get("parent") == parent and (name is None or f["name"] == name)
                and ("mimeType = " not in q or f.get("mimeType") == drive_api.FOLDER_MIME)
                and ("mimeType != " not in q or f.get("mimeType") != drive_api.FOLDER_MIME)]
        return _Req({"files": hits})

    def create(self, body, fields, media_body=None):
        fid = f"id{len(self.store.files) + 1}"
        if media_body is None:
            self.store.files[fid] = {"id": fid, "name": body["name"], "mimeType": body["mimeType"], "parent": body["parents"][0]}
            return _Req({"id": fid})
        return _Upload(self.store, fid, body["name"], body["parents"][0], media_body)

    def update(self, fileId, media_body, fields):
        f = self.store.files[fileId]
        return _Upload(self.store, fileId, f["name"], f["parent"], media_body)

    def get_media(self, fileId):
        return _Req(self.store.content[fileId])


class FakeStore:
    def __init__(self):
        self.files = {}
        self.content = {}
        self.fail_first = False


@pytest.fixture
def svc(monkeypatch):
    """A fake service plus a stand-in googleapiclient package (MediaFileUpload / HttpError) in sys.modules."""
    store = FakeStore()
    http = types.ModuleType("googleapiclient.http"); http.MediaFileUpload = MediaFileUpload
    errors = types.ModuleType("googleapiclient.errors"); errors.HttpError = HttpError
    pkg = types.ModuleType("googleapiclient"); pkg.http = http; pkg.errors = errors
    for name, mod in (("googleapiclient", pkg), ("googleapiclient.http", http), ("googleapiclient.errors", errors)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(drive_api.time, "sleep", lambda s: None)
    return types.SimpleNamespace(files=lambda: FakeFiles(store), store=store)


def test_resolve_create_upload_replace_and_list(tmp_path, svc):
    with pytest.raises(FileNotFoundError):
        drive_api.resolve_path(svc, ["GridNet", "cache2d"])
    folder = drive_api.resolve_path(svc, ["GridNet", "cache2d", "run"], create=True)
    assert drive_api.resolve_path(svc, ["GridNet", "cache2d", "run"]) == folder      # idempotent
    f = tmp_path / "zz.tar"; f.write_bytes(b"x" * 1000)
    seen = []
    r = drive_api.upload_file(svc, str(f), folder, chunk_mb=1, progress=lambda done, total: seen.append((done, total)))
    assert int(r["size"]) == 1000 and seen and seen[-1][1] == 1000 and r["seconds"] >= 0
    assert [x["name"] for x in drive_api.list_folder(svc, folder, suffix=".tar")] == ["zz.tar"]
    first_id = r["id"]
    f.write_bytes(b"y" * 2000)                                                     # replace keeps the file id
    r2 = drive_api.upload_file(svc, str(f), folder)
    assert r2["id"] == first_id and int(r2["size"]) == 2000
    assert len(drive_api.list_folder(svc, folder)) == 1
    assert drive_api.find_child(svc, folder, "zz.tar", folder=False)["id"] == first_id
    assert drive_api.find_child(svc, folder, "zz.tar", folder=True) is None


def test_upload_retries_transient_http_errors(tmp_path, svc):
    svc.store.fail_first = True
    folder = drive_api.resolve_path(svc, ["GridNet"], create=True)
    f = tmp_path / "a.tar"; f.write_bytes(b"z" * 10)
    r = drive_api.upload_file(svc, str(f), folder, chunk_mb=1)
    assert int(r["size"]) == 10


def test_zone_tar_on_drive(tmp_path, svc):
    import json
    folder = drive_api.resolve_path(svc, ["GridNet", "cache2d", "run"], create=True)
    assert drive_api.zone_tar_on_drive(svc, folder, "t1") is None
    tar = tmp_path / "t1.tar"; tar.write_bytes(b"t" * 5000)
    man = tmp_path / "t1.tar.json"; man.write_text(json.dumps({"zone": "t1", "n": 2, "ids": ["a", "b"], "bytes": 5000}))
    drive_api.upload_file(svc, str(tar), folder, mimetype="application/x-tar")
    assert drive_api.zone_tar_on_drive(svc, folder, "t1") is None            # manifest not uploaded yet
    drive_api.upload_file(svc, str(man), folder, mimetype="application/json")
    assert drive_api.zone_tar_on_drive(svc, folder, "t1")["n"] == 2
    tar.write_bytes(b"t" * 4000); drive_api.upload_file(svc, str(tar), folder)   # size no longer matches
    assert drive_api.zone_tar_on_drive(svc, folder, "t1") is None
