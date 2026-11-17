"""Minimal Google Drive API v3 helpers for Colab: upload big files WITHOUT going through the Drive mount.

The mount (FUSE) is convenient for reading, but writing a multi-GB file through it keeps gigabytes of dirty pages
and an upload buffer in the VM, and on a 12.7 GB runtime the Drive client gets OOM-killed ("Transport endpoint is
not connected"). A resumable API upload streams the file in fixed chunks instead (default 32 MB) and is independent
of the mount. Used by notebooks/colab_compact_cache2d.ipynb for the zone tars.

In Colab:
    from google.colab import auth; auth.authenticate_user()          # once per session (popup)
    svc = drive_service()
    folder = resolve_path(svc, ["GridNet", "cache2d", "upernet_convnext_t_v2"])
    upload_file(svc, "/content/x/t1z6a.tar", folder)
"""
import os
import time
from typing import Callable, List, Optional

FOLDER_MIME = "application/vnd.google-apps.folder"


def drive_service():
    from googleapiclient.discovery import build
    return build("drive", "v3", cache_discovery=False)


def _escape(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


def find_child(svc, parent_id: str, name: str, folder: Optional[bool] = None) -> Optional[dict]:
    """The first non-trashed child of parent_id called `name` ({id, name, size, mimeType}) or None."""
    q = f"name = '{_escape(name)}' and '{parent_id}' in parents and trashed = false"
    if folder is True:
        q += f" and mimeType = '{FOLDER_MIME}'"
    elif folder is False:
        q += f" and mimeType != '{FOLDER_MIME}'"
    res = svc.files().list(q=q, fields="files(id, name, size, mimeType)", pageSize=5, spaces="drive").execute()
    files = res.get("files", [])
    return files[0] if files else None


def resolve_path(svc, parts: List[str], root: str = "root", create: bool = False) -> str:
    """Folder id of My Drive/<parts[0]>/<parts[1]>/..., optionally creating missing folders."""
    cur = root
    for part in parts:
        hit = find_child(svc, cur, part, folder=True)
        if hit is None:
            if not create:
                raise FileNotFoundError(f"no folder '{part}' under id {cur} (path {'/'.join(parts)})")
            hit = svc.files().create(body={"name": part, "mimeType": FOLDER_MIME, "parents": [cur]}, fields="id").execute()
        cur = hit["id"]
    return cur


def upload_file(svc, path: str, folder_id: str, name: Optional[str] = None, chunk_mb: int = 32,
                mimetype: str = "application/octet-stream", replace: bool = True,
                progress: Optional[Callable[[int, int], None]] = None, retries: int = 5) -> dict:
    """Resumable upload of a local file into folder_id; an existing file of the same name is replaced in place
    (same file id) when replace=True. Returns {id, name, size} and checks that size matches the local file."""
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    name = name or os.path.basename(path)
    local_size = os.path.getsize(path)
    media = MediaFileUpload(path, mimetype=mimetype, resumable=True, chunksize=chunk_mb * 1024 * 1024)
    existing = find_child(svc, folder_id, name, folder=False) if replace else None
    if existing is not None:
        req = svc.files().update(fileId=existing["id"], media_body=media, fields="id, name, size")
    else:
        req = svc.files().create(body={"name": name, "parents": [folder_id]}, media_body=media, fields="id, name, size")
    resp, t0 = None, time.time()
    while resp is None:
        for attempt in range(retries):
            try:
                status, resp = req.next_chunk()
                break
            except HttpError as e:                       # 5xx / 429: back off and resume the same chunk
                if attempt == retries - 1 or (e.resp.status not in (429, 500, 502, 503, 504)):
                    raise
                time.sleep(2 ** attempt * 5)
        if status is not None and progress is not None:
            progress(status.resumable_progress, local_size)
    size = int(resp.get("size", -1))
    if size != local_size:
        raise IOError(f"upload of {name} ended with size {size}, local {local_size}")
    resp["seconds"] = time.time() - t0
    return resp


def list_folder(svc, folder_id: str, suffix: Optional[str] = None) -> List[dict]:
    """All non-trashed files directly under folder_id ({id, name, size}), optionally filtered by name suffix."""
    out, token = [], None
    while True:
        res = svc.files().list(q=f"'{folder_id}' in parents and trashed = false", fields="nextPageToken, files(id, name, size)",
                               pageSize=1000, pageToken=token, spaces="drive").execute()
        out += [f for f in res.get("files", []) if suffix is None or f["name"].endswith(suffix)]
        token = res.get("nextPageToken")
        if not token:
            return out


def read_small_file(svc, file_id: str) -> bytes:
    """Content of a (small) file by id."""
    return svc.files().get_media(fileId=file_id).execute()


def zone_tar_on_drive(svc, folder_id: str, zone: str) -> Optional[dict]:
    """The manifest of <zone>.tar in folder_id if both the tar and <zone>.tar.json are there and the tar has the
    size the manifest records (checked through the API, independent of the mount); else None."""
    import json
    tar = find_child(svc, folder_id, zone + ".tar", folder=False)
    man = find_child(svc, folder_id, zone + ".tar.json", folder=False)
    if tar is None or man is None:
        return None
    manifest = json.loads(read_small_file(svc, man["id"]).decode())
    return manifest if int(tar.get("size", -1)) == manifest.get("bytes") else None
