import os, sys, time, json, hashlib, re
from urllib.parse import urlparse, unquote
import requests

URL = os.environ["SRC_URL"]
NAME = os.environ.get("FILE_NAME", "").strip()
FOLDER = os.environ.get("FOLDER_ID", "").strip()
CHUNK = 256 * 1024 * 256          # 64 MiB (must be a multiple of 256 KiB)
BUDGET = 5 * 3600 + 40 * 60       # stop after 5h40m and restart
START = time.time()
STATE_NAME = ".transfer_" + hashlib.sha1(URL.encode()).hexdigest()[:12] + ".json"
API = "https://www.googleapis.com"


def log(m):
    print(m, flush=True)


class Auth:
    def __init__(self):
        self.tok, self.exp = None, 0

    def h(self):
        if time.time() > self.exp - 120:
            r = requests.post("https://oauth2.googleapis.com/token", data={
                "client_id": os.environ["GDRIVE_CLIENT_ID"],
                "client_secret": os.environ["GDRIVE_CLIENT_SECRET"],
                "refresh_token": os.environ["GDRIVE_REFRESH_TOKEN"],
                "grant_type": "refresh_token"}, timeout=30)
            r.raise_for_status()
            j = r.json()
            self.tok, self.exp = j["access_token"], time.time() + j["expires_in"]
        return {"Authorization": "Bearer " + self.tok}


auth = Auth()


def file_name():
    n = NAME or unquote(os.path.basename(urlparse(URL).path)) or "video.mp4"
    return re.sub(r'[\\/:*?"<>|]', "_", n)


UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def probe():
    r = requests.get(URL, headers={**UA, "Range": "bytes=0-0"}, stream=True, timeout=60)
    code, cr = r.status_code, r.headers.get("Content-Range", "")
    log(f"Source check: HTTP {code} | Accept-Ranges={r.headers.get('Accept-Ranges')} "
        f"| Content-Range={cr or '-'} | Content-Length={r.headers.get('Content-Length')}")
    r.close()
    if code != 206 or "/" not in cr:
        sys.exit(f"Source did not give a partial response (HTTP {code}). Cannot continue.")
    return int(cr.split("/")[1])


def find_state():
    parent = FOLDER or "root"
    q = f"name='{STATE_NAME}' and '{parent}' in parents and trashed=false"
    r = requests.get(f"{API}/drive/v3/files", headers=auth.h(),
                     params={"q": q, "fields": "files(id)"}, timeout=30)
    r.raise_for_status()
    files = r.json()["files"]
    if not files:
        return None, None
    fid = files[0]["id"]
    c = requests.get(f"{API}/drive/v3/files/{fid}?alt=media", headers=auth.h(), timeout=30)
    c.raise_for_status()
    return fid, c.json()


def save_state(fid, data):
    if fid:
        r = requests.patch(f"{API}/upload/drive/v3/files/{fid}?uploadType=media",
                           headers={**auth.h(), "Content-Type": "application/json"},
                           data=json.dumps(data), timeout=30)
        r.raise_for_status()
        return fid
    meta = {"name": STATE_NAME, "mimeType": "application/json"}
    if FOLDER:
        meta["parents"] = [FOLDER]
    b = "xxtransferxx"
    body = (f"--{b}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{json.dumps(meta)}\r\n"
            f"--{b}\r\nContent-Type: application/json\r\n\r\n{json.dumps(data)}\r\n--{b}--").encode()
    r = requests.post(f"{API}/upload/drive/v3/files?uploadType=multipart&fields=id",
                      headers={**auth.h(), "Content-Type": f"multipart/related; boundary={b}"},
                      data=body, timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def new_session(total, name, mime):
    meta = {"name": name}
    if FOLDER:
        meta["parents"] = [FOLDER]
    r = requests.post(f"{API}/upload/drive/v3/files?uploadType=resumable",
                      headers={**auth.h(), "Content-Type": "application/json; charset=UTF-8",
                               "X-Upload-Content-Length": str(total),
                               "X-Upload-Content-Type": mime},
                      data=json.dumps(meta), timeout=60)
    r.raise_for_status()
    return r.headers["Location"]


def query_offset(session, total):
    """Returns bytes already stored in Drive, or None if session is dead."""
    r = requests.put(session, headers={"Content-Length": "0",
                                       "Content-Range": f"bytes */{total}"}, timeout=60)
    if r.status_code in (200, 201):
        return total
    if r.status_code == 308:
        rng = r.headers.get("Range")
        return int(rng.split("-")[1]) + 1 if rng else 0
    if r.status_code in (404, 410):
        return None
    r.raise_for_status()


def fetch(a, b):
    for i in range(6):
        try:
            r = requests.get(URL, headers={**UA, "Range": f"bytes={a}-{b}"}, timeout=(30, 120))
            if r.status_code == 206 and len(r.content) == b - a + 1:
                return r.content
            log(f"source returned status {r.status_code}, retrying")
        except requests.RequestException as e:
            log(f"source error: {type(e).__name__}, retrying")
        time.sleep(5 * (i + 1))
    sys.exit("Source download failed repeatedly (link may have expired).")


def upload_chunk(session, data, start, total):
    end = start + len(data) - 1
    for i in range(6):
        try:
            r = requests.put(session, data=data, timeout=600, headers={
                "Content-Range": f"bytes {start}-{end}/{total}"})
            if r.status_code in (200, 201):
                return total
            if r.status_code == 308:
                rng = r.headers.get("Range")
                return int(rng.split("-")[1]) + 1 if rng else 0
            log(f"drive returned status {r.status_code}, retrying")
        except requests.RequestException as e:
            log(f"drive error: {type(e).__name__}, retrying")
        time.sleep(5 * (i + 1))
        off = query_offset(session, total)
        if off is not None and off > start:
            return off
    sys.exit("Drive upload failed repeatedly.")


def main():
    name = file_name()
    mime = "video/x-matroska" if name.lower().endswith(".mkv") else "video/mp4"
    total = probe()
    log(f"File: {name} | size: {total / 2**30:.2f} GB")

    fid, st = find_state()
    session, offset = None, 0
    if st and st.get("total") == total:
        off = query_offset(st["session"], total)
        if off is not None:
            session, offset = st["session"], off
            log(f"Resuming from {offset / 2**30:.2f} GB")
    if session is None:
        session = new_session(total, name, mime)
        fid = save_state(fid, {"session": session, "total": total, "name": name})
        log("Started new upload session")

    while offset < total:
        if time.time() - START > BUDGET:
            log("Time budget reached, scheduling continuation")
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("again=true\n")
            return
        end = min(offset + CHUNK, total) - 1
        data = fetch(offset, end)
        offset = upload_chunk(session, data, offset, total)
        log(f"{offset / total * 100:.1f}%  ({offset / 2**30:.2f} / {total / 2**30:.2f} GB)")

    if fid:
        requests.delete(f"{API}/drive/v3/files/{fid}", headers=auth.h(), timeout=30)
    log("DONE: file is in Google Drive")


if __name__ == "__main__":
    main()
