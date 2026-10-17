"""
Movie Bot: Drive video -> multi-resolution -> Internet Archive (custom player + downloads, unlisted) + screenshots + 2:3 poster thumbnail
-> Gemini title/description/labels -> Blogger post (published by default).
Runs on GitHub Actions. All settings come from environment variables.

Screenshots and the poster thumbnail are exported as high-quality WebP.
Converted movie files are uploaded to Internet Archive (noindex) and the
Download buttons in the post link directly to archive.org/download/...

Thumbnail priority:
  1. TMDB official poster(s)   (best 2:3 poster, highest vote, original size)
  2. OMDb / IMDb poster
  3. Gemini + Google Image Search poster
  4. 2:3 crop from the movie source frame (last fallback)
"""
import hashlib
import difflib
import base64
import html
import http.client
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from internetarchive import get_item as ia_get_item
import requests

# ---------- settings ----------
# Drive now authenticates via a service account (never expires).
# Blogger still needs a normal user OAuth refresh token, since Blogger
# does not support service-account access to a personal blog.
SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["GOOGLE_REFRESH_TOKEN"]
# GEMINI_API_KEY may hold ONE key or MANY keys separated by commas.
# When a key hits its quota (HTTP 429), the bot automatically switches to the
# next key. Every Gemini call in this file is covered by the patch below.
_GEMINI_KEYS = [k.strip() for k in os.environ["GEMINI_API_KEY"].replace("\n", ",").split(",") if k.strip()]
if not _GEMINI_KEYS:
    raise RuntimeError("GEMINI_API_KEY secret is empty.")
GEMINI_KEY = _GEMINI_KEYS[0]
_GEMINI_DEAD = set()  # (key, model-path) pairs that are out of quota this run
_ORIG_URLOPEN = urllib.request.urlopen


def _gemini_urlopen(url, *args, **kwargs):
    """Drop-in urlopen: for Gemini URLs, rotate to the next API key on 429/403."""
    import io
    req = url if isinstance(url, urllib.request.Request) else None
    full = req.full_url if req is not None else str(url)
    if "generativelanguage.googleapis.com" not in full or "key=" not in full:
        return _ORIG_URLOPEN(url, *args, **kwargs)

    parsed = urllib.parse.urlsplit(full)
    model_path = parsed.path
    params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    last_err = None
    for idx, key in enumerate(_GEMINI_KEYS, 1):
        if (key, model_path) in _GEMINI_DEAD:
            continue
        new_params = [(k, key if k == "key" else v) for k, v in params]
        new_url = urllib.parse.urlunsplit(
            parsed._replace(query=urllib.parse.urlencode(new_params)))
        if req is not None:
            new_req = urllib.request.Request(
                new_url, data=req.data, headers=dict(req.header_items()),
                method=req.get_method())
        else:
            new_req = new_url
        try:
            return _ORIG_URLOPEN(new_req, *args, **kwargs)
        except urllib.error.HTTPError as ex:
            if ex.code in (429, 403):
                _GEMINI_DEAD.add((key, model_path))
                last_err = ex
                print(f"  Gemini key #{idx}/{len(_GEMINI_KEYS)} HTTP {ex.code} -> trying next key",
                      flush=True)
                continue
            raise
    if last_err is not None:
        raise last_err
    body = io.BytesIO(b'{"error":{"code":429,"message":"All Gemini API keys are exhausted for this model."}}')
    raise urllib.error.HTTPError(full, 429, "All Gemini keys exhausted", {}, body)


urllib.request.urlopen = _gemini_urlopen
BLOG_ID = os.environ["BLOG_ID"]
INPUT_FOLDER = os.environ.get("DRIVE_INPUT_FOLDER_ID", "").strip()
# Multi-account mode: ACCOUNTS_JSON = JSON list of Drive accounts
# [{"name","client_id","client_secret","refresh_token"}, ...]. When set, the bot
# looks for a folder called MULTI_FOLDER_NAME in EVERY account and processes the
# videos it finds there. When empty, the old single-folder behaviour is used.
ACCOUNTS_JSON = os.environ.get("ACCOUNTS_JSON", "").strip()
MULTI_FOLDER_NAME = os.environ.get("MULTI_FOLDER_NAME", "movie-input").strip() or "movie-input"
if not ACCOUNTS_JSON and not INPUT_FOLDER:
    raise RuntimeError("Set ACCOUNTS_JSON (multi-account) or DRIVE_INPUT_FOLDER_ID (single folder).")

# Internet Archive S3-style keys: https://archive.org/account/s3.php
# Store them in GitHub Actions Secrets as IA_ACCESS and IA_SECRET.
IA_ACCESS = os.environ["IA_ACCESS"].strip()
IA_SECRET = os.environ["IA_SECRET"].strip()
# File type of the DOWNLOAD button (default mkv, so browsers download it
# instead of playing it). The player always uses a separate MP4 file.
IA_FILE_EXT = os.environ.get("IA_FILE_EXT", "mkv").strip(". ").lower() or "mkv"

# ShrtFly URL shortener: Download buttons in the Blogger post use the
# shortened link instead of the direct archive.org link.
# Get the API key from shrtfly.com -> Tools -> Developers API.
# Store it in GitHub Actions Secrets as SHRTFLY_API_KEY.
SHRTFLY_API_KEY = os.environ.get("SHRTFLY_API_KEY", "").strip()
SHRTFLY_API_URL = os.environ.get("SHRTFLY_API_URL", "https://shrtfly.com/api").strip()
# true  = if shortening fails, stop (state is saved, next run retries) so a
#         direct archive.org link is NEVER published.
# false = if shortening fails, fall back to the direct link.
SHRTFLY_STRICT = os.environ.get("SHRTFLY_STRICT", "true").lower() == "true"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def _gem_models():
    """Primary model + fallbacks (used when Google returns 503 'high demand')."""
    cand = [GEMINI_MODEL,
            os.environ.get("ANALYSIS_GEMINI_FALLBACK", "gemini-3.5-flash"),
            os.environ.get("ANALYSIS_GEMINI_FALLBACK2", "gemini-2.5-flash")]
    return [m for m in dict.fromkeys(cand) if m]

# Watch Online (player Quality menu): 180p .. 1080p.
PLAY_RESOLUTIONS = [int(x) for x in os.environ.get(
    "PLAY_RESOLUTIONS", "180,240,360,480,720,1080").split(",") if x.strip()]
# Download buttons: 720p + 1080p, plus 1440p (2K) / 2160p (4K) when the source
# really is that big. A size larger than the source is never made.
DOWNLOAD_RESOLUTIONS = [int(x) for x in os.environ.get(
    "DOWNLOAD_RESOLUTIONS", "720,1080,1440,2160").split(",") if x.strip()]
MAX_VIDEOS = int(os.environ.get("MAX_VIDEOS", "1"))
PUBLISH = os.environ.get("PUBLISH", "true").lower() == "true"
LANGUAGE_HINT = os.environ.get("LANGUAGE_HINT", "").strip()
AUDIO_MINUTES = int(os.environ.get("AUDIO_MINUTES", "10"))
SCREENSHOTS = int(os.environ.get("SCREENSHOTS", "6"))
WAIT_SECONDS = int(os.environ.get("WAIT_SECONDS", "20"))
DIRECTOR_NAME = os.environ.get("DIRECTOR_NAME", "").strip()
# Optional manual title. When set, the same title is used for the Blogger post,
# Movie Info -> Movie Name, thumbnail search, and synopsis context.
MANUAL_TITLE = next((
    os.environ.get(k, "").strip()
    for k in ("MANUAL_TITLE", "MOVIE_TITLE", "CUSTOM_TITLE")
    if os.environ.get(k, "").strip()
), "")
# Higher CRF = smaller file. These values keep outputs near or below a
# typical 2 GB source (the old 23/22 made 720p bigger than the original).
# QUALITY-FIRST values (lower CRF = better picture, bigger file). The Watch
# MP4 is also the base of the Download file, so both keep high quality.
CRF = {144: 28, 180: 27, 240: 26, 360: 24, 480: 23, 720: 21, 1080: 19, 1440: 20, 2160: 20}
# true  = when the source is already browser-friendly H.264 (8-bit) and the
#         requested size equals the source size, the video is COPIED untouched
#         (100% original quality, no re-encode, much faster).
PLAY_COPY_ORIGINAL = os.environ.get("PLAY_COPY_ORIGINAL", "true").lower() == "true"
X264_PRESET = os.environ.get("X264_PRESET", "faster")   # slower preset = smaller file, more time

# ---------- image quality (WebP) ----------
# Screenshots and the poster thumbnail are exported as WebP.
# Quality 0-100 (higher = sharper, bigger file). compression_level 0-6
# (6 = slowest, best compression for the same quality).
SHOT_WEBP_QUALITY = int(os.environ.get("SHOT_WEBP_QUALITY", "95"))
THUMB_WEBP_QUALITY = int(os.environ.get("THUMB_WEBP_QUALITY", "97"))
WEBP_COMPRESSION = int(os.environ.get("WEBP_COMPRESSION", "6"))
# Poster thumbnail size (exact 2:3). Raise THUMB_W (e.g. 1000 or 1200)
# for a sharper official poster; official posters are 2:3 so no real crop.
THUMB_W = int(os.environ.get("THUMB_W", "1000"))
THUMB_H = THUMB_W * 3 // 2

# ---------- TMDB (optional) ----------
# If neither value is set, the bot skips TMDB and works exactly as before.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()          # v3 key
TMDB_READ_TOKEN = os.environ.get("TMDB_READ_TOKEN", "").strip()    # v4 read token

# ---------- OMDb (optional) ----------
# Gives the real IMDb rating + basic facts. Free key: omdbapi.com/apikey.aspx
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "").strip()
# Optional manual IMDb id (e.g. tt0111161). If valid, it is used to match the
# exact movie on TMDB and OMDb instead of searching by name.
IMDB_ID = os.environ.get("IMDB_ID", "").strip()

# ---------- local AI fallback ----------
# Gemini remains the primary AI. Qwen3-VL is started ONLY when a Gemini
# task has failed completely. The local model runs through llama.cpp so no
# external AI API/file upload is needed for the fallback.
LOCAL_AI_ENABLED = os.environ.get("LOCAL_AI_ENABLED", "true").lower() == "true"
LOCAL_QWEN_MODEL = os.environ.get(
    "LOCAL_QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M"
)
LOCAL_AI_TIMEOUT = int(os.environ.get("LOCAL_AI_TIMEOUT", "900"))
LOCAL_AI_CONTEXT = int(os.environ.get("LOCAL_AI_CONTEXT", "8192"))
LOCAL_AI_THREADS = int(os.environ.get("LOCAL_AI_THREADS", "4"))
LOCAL_AI_MAX_IMAGES = int(os.environ.get("LOCAL_AI_MAX_IMAGES", "12"))


# After the Blogger post is created, the source video is MOVED from
# movie-input to the _processed folder (same Drive account). It is no longer
# deleted. (Old DELETE_DRIVE_VIDEO flag is ignored.)

WORK = Path("work")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
USER_SCOPES = [
    "https://www.googleapis.com/auth/blogger",
    "https://www.googleapis.com/auth/drive.file",
]


def log(*a):
    print(*a, flush=True)


def retry(fn, tries=4, base_delay=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa
            if i == tries - 1:
                raise
            delay = min(base_delay * (3 ** i), 120)
            log(f"  retry {i + 1} after error: {str(e)[:300]} (wait {delay}s)")
            time.sleep(delay)


def run(cmd):
    subprocess.run(cmd, check=True)


# ---------- resume / checkpoint ----------
# GitHub Actions kills a job after 6 hours and wipes the runner disk.
# After every finished step the progress is saved to state/<video_id>.json.
# The workflow commits that folder and re-triggers itself, so the next run
# skips finished steps and continues from where the last run stopped.
STATE_DIR = Path(os.environ.get("STATE_DIR", "state"))   # can point to another repo checkout
RUN_START = time.time()
# Soft limit: do not START a new big step if it probably can't finish in time.
MAX_RUN_MINUTES = int(os.environ.get("MAX_RUN_MINUTES", "330"))
RESUME_EXIT_CODE = 75


class ResumeLater(Exception):
    """Raised when time is nearly up; state is already saved."""


def need_time(minutes_needed):
    used = (time.time() - RUN_START) / 60
    if used > MAX_RUN_MINUTES - minutes_needed:
        raise ResumeLater(
            f"{used:.0f} min used; not enough time left for the next step")


# ---------- persistent cache for Gemini / OMDb / TMDB answers ----------
# Good answers are saved in state/cache/ (committed by the workflow), so the
# same movie is never looked up twice. Empty / failed answers are NOT cached.
CACHE_DIR = STATE_DIR / "cache"


def _is_good(result):
    if not result:
        return False
    if isinstance(result, str):
        return result.strip().upper() not in ("N/A", "NONE", "NULL", "UNKNOWN")
    return True


def disk_cache(name, ttl_days=14):
    def wrap(fn):
        def inner(*args, **kwargs):
            try:
                raw = json.dumps([args, kwargs], sort_keys=True, default=str, ensure_ascii=False).lower()
                key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
                path = CACHE_DIR / f"{name}-{key}.json"
            except Exception:  # noqa
                return fn(*args, **kwargs)
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                if time.time() - float(d.get("time", 0)) < ttl_days * 86400 and _is_good(d.get("result")):
                    log(f"  [cache] {name}: using saved result")
                    return d["result"]
            except Exception:  # noqa
                pass
            result = fn(*args, **kwargs)
            if _is_good(result):
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"time": time.time(), "result": result},
                                               ensure_ascii=False), encoding="utf-8")
                except Exception as e:  # noqa
                    log(f"  [cache] could not save {name}: {e}")
            return result
        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner
    return wrap


def _state_file(video_id):
    return STATE_DIR / f"{video_id}.json"


def load_state(video_id):
    try:
        return json.loads(_state_file(video_id).read_text(encoding="utf-8"))
    except Exception:  # noqa
        return {}


def save_state(video_id, st):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _state_file(video_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_state_file(video_id))


def clear_state(video_id):
    try:
        _state_file(video_id).unlink()
    except FileNotFoundError:
        pass


# ---------- Google clients ----------
# Drive (read side): service account credentials. These never expire on
# their own. Used for listing/downloading input videos, creating small
# folders, and moving files between folders -- none of this consumes
# storage quota, so the service account's lack of Drive storage is fine.
sa_info = json.loads(SERVICE_ACCOUNT_JSON)
drive_creds = service_account.Credentials.from_service_account_info(
    sa_info, scopes=DRIVE_SCOPES
)
drive = build("drive", "v3", credentials=drive_creds, cache_discovery=False)

# User OAuth (write side + Blogger): normal user refresh token, scoped to
# Blogger and drive.file. Service accounts have zero Drive storage quota,
# so NEW files (thumbnails, screenshots) must be created under the real
# Google account, which owns actual storage.
# drive.file is a "sensitive" (not "restricted") scope, so it doesn't
# block publishing the OAuth consent screen the way full "drive" does.
user_creds = Credentials(
    None,
    refresh_token=REFRESH_TOKEN,
    token_uri="https://oauth2.googleapis.com/token",
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    scopes=USER_SCOPES,
)
user_creds.refresh(Request())
blogger = build("blogger", "v3", credentials=user_creds, cache_discovery=False)


# ---------- Drive helpers ----------
# Multi-account support: every video remembers which account/folder it came from.
_FILE_CTX = {}   # file_id -> {"svc": drive client, "folder": input folder id, "acct": label}


def _acct_service(acc):
    creds = Credentials(
        None,
        refresh_token=acc["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=acc["client_id"],
        client_secret=acc["client_secret"],
        scopes=DRIVE_SCOPES,
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_input_folder(svc):
    """Existing MULTI_FOLDER_NAME folder (My Drive first). Never creates one."""
    base = (f"name='{MULTI_FOLDER_NAME}' and "
            "mimeType='application/vnd.google-apps.folder' and trashed=false")
    for extra in (" and 'root' in parents", ""):
        res = svc.files().list(q=base + extra, fields="files(id)",
                               orderBy="createdTime").execute()["files"]
        if res:
            return res[0]["id"]
    return None


def _drv(file_id):
    ctx = _FILE_CTX.get(file_id)
    return ctx["svc"] if ctx else drive


def _in_folder(file_id):
    ctx = _FILE_CTX.get(file_id)
    return ctx["folder"] if ctx else INPUT_FOLDER


def _ensure_child(svc, parent, name):
    q = (f"'{parent}' in parents and name='{name}' and "
         "mimeType='application/vnd.google-apps.folder' and trashed=false")
    res = svc.files().list(q=q, fields="files(id)").execute()
    if res["files"]:
        return res["files"][0]["id"]
    body = {"name": name, "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent]}
    return svc.files().create(body=body, fields="id").execute()["id"]


def ensure_folder(name):
    return _ensure_child(drive, INPUT_FOLDER, name)


def move_to_processed(file_id, processed_folder):
    """Move the source video from movie-input to that account's _processed
    folder (uses the existing _processed folder, creates it only if missing)."""
    svc, parent = _drv(file_id), _in_folder(file_id)
    pf = processed_folder or _ensure_child(svc, parent, "_processed")
    retry(lambda: svc.files().update(fileId=file_id, addParents=pf,
                                     removeParents=parent, fields="id",
                                     supportsAllDrives=True).execute())
    acct = (_FILE_CTX.get(file_id) or {}).get("acct", "")
    log(f"  Drive video moved to _processed{f' [{acct}]' if acct else ''}.")


def delete_drive_video(file_id, processed_folder):
    """Permanently delete the source video from Drive after a successful post.
    1) files.delete (permanent)  2) move to Trash  3) move to _processed.
    A missing file (404) counts as already deleted."""
    svc = _drv(file_id)
    try:
        svc.files().delete(fileId=file_id, supportsAllDrives=True).execute()
        log("  Drive video permanently deleted.")
        return "deleted"
    except Exception as ex:  # noqa
        if getattr(getattr(ex, "resp", None), "status", None) == 404:
            log("  Drive video already gone.")
            return "deleted"
        log(f"  Permanent delete failed ({str(ex)[:200]}); trying Trash...")
    try:
        svc.files().update(fileId=file_id, body={"trashed": True},
                           supportsAllDrives=True, fields="id").execute()
        log("  Drive video moved to Trash (account is not the owner, "
            "so it cannot delete permanently).")
        return "trashed"
    except Exception as ex:  # noqa
        log(f"  Trash failed ({str(ex)[:200]}); moving to _processed instead.")
    move_to_processed(file_id, processed_folder)
    return "moved"


# ---------- multi-bot claim (so two bots never take the same video) ----------
# Each bot writes a PUBLIC custom property on the Drive file (claimed_by,
# claim_ts). Public `properties` (not appProperties) are used because
# appProperties are private to one OAuth client and the other bot could not
# read them. BOT_ID defaults to the GitHub repo name, so two repos are
# automatically different bots without any extra setting.
BOT_ID = (os.environ.get("BOT_ID") or os.environ.get("GITHUB_REPOSITORY")
          or "bot1").strip()
CLAIM_TTL_HOURS = float(os.environ.get("CLAIM_TTL_HOURS", "24"))
CLAIM_SETTLE_SECONDS = float(os.environ.get("CLAIM_SETTLE_SECONDS", "5"))


def _claim_info(f):
    p = (f or {}).get("properties") or {}
    try:
        ts = float(p.get("claim_ts", "0"))
    except ValueError:
        ts = 0.0
    return p.get("claimed_by"), ts


def _claimed_by_other(f):
    by, ts = _claim_info(f)
    return bool(by) and by != BOT_ID and (time.time() - ts) < CLAIM_TTL_HOURS * 3600


def try_claim(f):
    """True if this bot now owns the video. False if another bot has it."""
    fid = f["id"]
    svc = _drv(fid)
    try:
        cur = svc.files().get(fileId=fid, fields="properties",
                              supportsAllDrives=True).execute()
        by, _ts = _claim_info(cur)
        if _claimed_by_other(cur):
            log(f"  '{f.get('name')}' is already taken by bot '{by}' - skipping.")
            return False
        stamp = {"claimed_by": BOT_ID, "claim_ts": str(int(time.time()))}
        svc.files().update(fileId=fid, body={"properties": stamp}, fields="id",
                           supportsAllDrives=True).execute()
        if by == BOT_ID:      # my own video (resume) - nothing to race against
            return True
        time.sleep(CLAIM_SETTLE_SECONDS)   # if two bots wrote at once, last write wins
        cur = svc.files().get(fileId=fid, fields="properties",
                              supportsAllDrives=True).execute()
        winner, _ = _claim_info(cur)
        if winner != BOT_ID:
            log(f"  '{f.get('name')}' was claimed by bot '{winner}' at the same "
                "moment - skipping.")
            return False
        log(f"  Claimed '{f.get('name')}' for bot '{BOT_ID}'.")
        return True
    except Exception as ex:  # noqa
        # Fail open: if the claim cannot be written, behave like before.
        log(f"  Claim check failed ({str(ex)[:200]}); continuing without a claim.")
        return True


def list_videos():
    if not ACCOUNTS_JSON:
        q = f"'{INPUT_FOLDER}' in parents and mimeType contains 'video/' and trashed=false"
        res = drive.files().list(q=q, fields="files(id,name,size,createdTime,properties,md5Checksum)",
                                 orderBy="createdTime").execute()
        return [f for f in res["files"] if not _claimed_by_other(f)]

    out = []
    for acc in json.loads(ACCOUNTS_JSON):
        label = acc.get("name", "?")
        try:
            svc = _acct_service(acc)
            fid = _find_input_folder(svc)
            if not fid:
                log(f"  [{label}] no '{MULTI_FOLDER_NAME}' folder - skipped")
                continue
            q = f"'{fid}' in parents and mimeType contains 'video/' and trashed=false"
            files = svc.files().list(q=q, fields="files(id,name,size,createdTime,properties,md5Checksum)",
                                     orderBy="createdTime", pageSize=100).execute()["files"]
            files = [f for f in files if not _claimed_by_other(f)]
        except Exception as ex:  # noqa
            log(f"  [{label}] skipped: {str(ex)[:200]}")
            continue
        log(f"  [{label}] {len(files)} video(s) found")
        for f in files:
            _FILE_CTX[f["id"]] = {"svc": svc, "folder": fid, "acct": label}
            out.append(f)
    # Videos that are already in progress (have a saved state) go FIRST, so a
    # long movie that needs several runs is always resumed before a new one starts.
    out.sort(key=lambda f: (0 if (_state_file(f["id"]).exists()
                                  or _claim_info(f)[0] == BOT_ID) else 1,
                            f.get("createdTime", "")))
    return out


def download(file_id, dest):
    req = _drv(file_id).files().get_media(fileId=file_id)
    with open(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req, chunksize=64 * 1024 * 1024)
        done = False
        while not done:
            status, done = retry(dl.next_chunk)
            if status:
                log(f"  download {int(status.progress() * 100)}%")


# ---------- duplicate guard ----------
# If the SAME movie is uploaded to movie-input again by mistake, the bot does
# not process it again. Every finished video is written to a permanent history
# stored in a hidden DRAFT PAGE on the Blogger blog. Both bots use the same blog,
# so they automatically read and write the SAME history (no setup, no GitHub
# state, no Drive/Archive storage). It still works after you delete the old video
# from _processed. A backup copy is kept in state/processed_videos.json.
# A new file is compared with the history (Drive md5 checksum, or same name +
# same size) and also with whatever is currently in _processed. On a match it is
# moved to _processed renamed "DUPLICATE - <name>". DEDUP=false switches it off.
# The draft page is not public. Do not delete it (if you do, history restarts).
HISTORY_PAGE_TITLE = "CINEMAWALA bot history (do not delete)"
HISTORY_FILE = STATE_DIR / "processed_videos.json"
DEDUP = os.environ.get("DEDUP", "true").lower() == "true"
_PROCESSED_FP = None   # fingerprints of everything already processed
_BLOG_HIST_CACHE = None


def _fp_keys(f):
    keys = set()
    md5 = f.get("md5Checksum")
    if md5:
        keys.add(("md5", md5))
    size = f.get("size")
    if size and f.get("name"):
        keys.add(("ns", str(f["name"]).strip().lower(), str(size)))
    return keys


def _keys_of(items):
    out = set()
    for h in items:
        out |= _fp_keys(h)
    return out


def _merge_history(*lists):
    merged, seen = [], set()
    for lst in lists:
        for h in lst:
            keys = _fp_keys(h)
            if keys and not (keys & seen):
                seen |= keys
                merged.append(h)
    return merged


def _history_local():
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa
        return []


def _blogger_history_read():
    """Read the history from the hidden Blogger page(s).
    Returns (items, ok, page_id). ok=False means 'could not read it safely'
    (then nothing is ever written online, so the history cannot be damaged)."""
    def fetch():
        pages, token = [], None
        while True:
            kw = dict(blogId=BLOG_ID, fetchBodies=True, maxResults=100,
                      status=["draft", "live"])
            if token:
                kw["pageToken"] = token
            res = blogger.pages().list(**kw).execute()
            pages += res.get("items", [])
            token = res.get("nextPageToken")
            if not token:
                return pages

    try:
        pages = retry(fetch, tries=3, base_delay=5)
    except Exception as ex:  # noqa
        log(f"  Could not read Blogger history page ({str(ex)[:200]}).")
        return [], False, None
    mine = sorted((p for p in pages if p.get("title") == HISTORY_PAGE_TITLE),
                  key=lambda p: str(p.get("id")))
    lists, ok = [], True
    for pg in mine:
        try:
            m = re.search(r"<pre[^>]*>(.*?)</pre>", pg.get("content", ""), re.S)
            data = json.loads(html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))) if m else []
            lists.append(data if isinstance(data, list) else [])
        except Exception as ex:  # noqa
            log(f"  Blogger history page is unreadable ({str(ex)[:150]}).")
            ok = False
    return _merge_history(*lists), ok, (mine[0]["id"] if mine else None)


def _blogger_history_write(items, page_id):
    body = {"kind": "blogger#page", "title": HISTORY_PAGE_TITLE,
            "content": "<p>Do not delete: duplicate-video history used by the movie bots.</p>"
                       "<pre>" + html.escape(json.dumps(
                           items, separators=(",", ":"), ensure_ascii=False)) + "</pre>"}
    if page_id:
        retry(lambda: blogger.pages().update(
            blogId=BLOG_ID, pageId=page_id, body=body).execute())
    else:
        retry(lambda: blogger.pages().insert(
            blogId=BLOG_ID, body=body, isDraft=True).execute())


def record_processed(video):
    """Remember a finished video forever (Blogger draft page + state backup)."""
    global _BLOG_HIST_CACHE
    item = {"name": video.get("name"), "size": video.get("size"),
            "md5Checksum": video.get("md5Checksum"),
            "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
    remote, ok, page_id = _blogger_history_read()      # always fresh before writing
    hist = _merge_history(_history_local(), remote)
    if not (_fp_keys(item) & _keys_of(hist)):
        hist.append(item)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = HISTORY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(HISTORY_FILE)
    except Exception as ex:  # noqa
        log(f"  Could not save local processed history ({str(ex)[:200]}).")
    if not ok:
        log("  Blogger history could not be read safely - NOT writing (local copy only).")
        return
    try:
        for attempt in range(1, 4):
            if len(_keys_of(hist)) <= len(_keys_of(remote)) and _keys_of(hist) <= _keys_of(remote):
                break                      # online copy already has everything
            _blogger_history_write(hist, page_id)
            time.sleep(3)
            remote, ok, page_id = _blogger_history_read()   # verify (the other bot may
            if not ok:                                     # have written at the same time)
                break
            if page_id is None:
                log("  History page was created but cannot be found again - stopping.")
                break
            if _keys_of(hist) <= _keys_of(remote):
                _BLOG_HIST_CACHE = remote
                log(f"  Saved to Blogger history page ({len(remote)} movie(s) recorded).")
                break
            hist = _merge_history(hist, remote)
            log(f"  Other bot wrote at the same time - merging and retrying ({attempt}/3).")
    except Exception as ex:  # noqa
        log(f"  Could not save Blogger history ({str(ex)[:200]}); local copy only.")


def _load_processed_fingerprints():
    global _PROCESSED_FP, _BLOG_HIST_CACHE
    if _PROCESSED_FP is not None:
        return _PROCESSED_FP
    fps = _keys_of(_history_local())
    if _BLOG_HIST_CACHE is None:
        _BLOG_HIST_CACHE = _blogger_history_read()[0]
    fps |= _keys_of(_BLOG_HIST_CACHE)
    targets = {}
    if _FILE_CTX:
        for ctx in _FILE_CTX.values():
            targets[(ctx["acct"], ctx["folder"])] = (ctx["svc"], ctx["folder"])
    else:
        targets[("single", INPUT_FOLDER)] = (drive, INPUT_FOLDER)
    for svc, parent in targets.values():
        try:
            q = (f"'{parent}' in parents and name='_processed' and "
                 "mimeType='application/vnd.google-apps.folder' and trashed=false")
            for fo in svc.files().list(q=q, fields="files(id)").execute()["files"]:
                token = None
                while True:
                    res = svc.files().list(
                        q=f"'{fo['id']}' in parents and trashed=false",
                        fields="nextPageToken,files(name,size,md5Checksum)",
                        pageSize=1000, pageToken=token).execute()
                    for pf in res.get("files", []):
                        fps |= _fp_keys(pf)
                    token = res.get("nextPageToken")
                    if not token:
                        break
        except Exception as ex:  # noqa
            log(f"  Duplicate check: could not read _processed ({str(ex)[:200]}).")
    log(f"  Duplicate check: {len(fps)} fingerprint(s) loaded (Blogger history + local + _processed).")
    _PROCESSED_FP = fps
    return fps


def is_duplicate(f):
    """True if this video was already processed before (same movie re-uploaded)."""
    if not DEDUP:
        return False
    fid = f["id"]
    if _state_file(fid).exists() or _claim_info(f)[0] == BOT_ID:
        return False          # my own video, already in progress - not a duplicate
    return bool(_fp_keys(f) & _load_processed_fingerprints())


def move_duplicate(f, processed_folder):
    """Move a duplicate to _processed (renamed) without processing it."""
    fid = f["id"]
    svc, parent = _drv(fid), _in_folder(fid)
    pf = processed_folder or _ensure_child(svc, parent, "_processed")
    body = {"name": f"DUPLICATE - {f.get('name', 'video')}"}
    retry(lambda: svc.files().update(fileId=fid, body=body, addParents=pf,
                                     removeParents=parent, fields="id",
                                     supportsAllDrives=True).execute())
    log(f"  Duplicate '{f.get('name')}' moved to _processed (not processed again).")


# ---------- Internet Archive helpers ----------
def make_ia_item_id(slug):
    """Unguessable item id, e.g. my-movie-3fa91c27b0d4 (one item per movie)."""
    base = re.sub(r"[^a-z0-9-]+", "-", str(slug).lower()).strip("-") or "movie"
    return f"{base[:60].strip('-') or 'movie'}-{secrets.token_hex(6)}"


# ---------- Image SEO helpers ----------
def _seo_title_year(title, year):
    """Clean title (no trailing '(2014)') and the 4-digit year, if any."""
    t = re.sub(r"\s*\(\s*\d{4}\s*\)\s*$", "", str(title or "")).strip()
    t = t.split("|")[0].strip()
    y = str(year or "").strip()
    if not re.fullmatch(r"\d{4}", y):
        m = re.search(r"\b(19|20)\d{2}\b", str(title or ""))
        y = m.group(0) if m else ""
    return t, y


def seo_image_base(meta, fallback_slug="movie"):
    """'Interstellar' + 2014 -> 'interstellar-2014' (used for image filenames)."""
    import unicodedata
    raw = MANUAL_TITLE or meta.get("title") or ""
    t, y = _seo_title_year(raw, meta.get("release_year"))
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:60].strip("-")
    base = base or re.sub(r"[^a-z0-9]+", "-", str(fallback_slug).lower()).strip("-") or "movie"
    return f"{base}-{y}" if y else base


def seo_rename(path, new_stem):
    """Rename a file to <new_stem><same extension> and return the new path."""
    p = Path(path)
    dst = p.with_name(new_stem + p.suffix.lower())
    if dst != p:
        if dst.exists():
            dst.unlink()
        p.rename(dst)
    return str(dst)


def seo_alt(meta, kind):
    """'Interstellar 2014 movie poster' / 'Interstellar 2014 movie screenshot 3'."""
    t, y = _seo_title_year(MANUAL_TITLE or meta.get("title") or "", meta.get("release_year"))
    return " ".join(x for x in (t or "Movie", y, kind) if x)


def ia_upload_file(path, item_id, title):
    """
    Upload one file to an Internet Archive item and return its permanent
    direct download URL. The item is marked noindex so it does not show up
    in archive.org search/browse (only people with the link can find it).
    """
    def do_upload():
        # The default metadata timeout of the library is only 12 s, which
        # often fails on GitHub runners. Use a longer connect/read timeout.
        item = ia_get_item(item_id, request_kwargs={"timeout": (60, 120)})
        return item.upload(
            [path],
            metadata={
                "mediatype": "data",   # no video player / derivatives
                "title": str(title)[:200],
                "noindex": "true",
            },
            access_key=IA_ACCESS,
            secret_key=IA_SECRET,
            queue_derive=False,
            checksum=True,   # skip files already uploaded (resume-safe)
            retries=8,
            retries_sleep=30,
            verbose=True,
        )

    try:
        res = retry(do_upload, tries=6, base_delay=20)
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout) as ex:
        # archive.org is unreachable right now. Progress is saved, so wait
        # a bit and let the workflow continue this video in a new run.
        log(f"  archive.org not reachable ({str(ex)[:200]}). Waiting 3 min, "
            "then resuming in a new run...")
        time.sleep(180)
        raise ResumeLater("archive.org unreachable")
    if not res or not all(getattr(r, "status_code", 200) in (200, None, "skipped") for r in res):
        raise RuntimeError(f"Internet Archive upload failed: {res}")

    url = ("https://archive.org/download/" + item_id + "/"
           + urllib.parse.quote(os.path.basename(path)))
    log("  Internet Archive link:", url)
    return url


# ---------- ShrtFly URL shortener ----------
def shorten_url(long_url):
    """Return the ShrtFly short link for long_url (raises on failure)."""
    api = (f"{SHRTFLY_API_URL}?api={urllib.parse.quote(SHRTFLY_API_KEY, safe='')}"
           f"&url={urllib.parse.quote(long_url, safe='')}&type=1&format=json")

    def call():
        req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace").strip()
        try:
            data = json.loads(raw)
        except ValueError:
            data = {}
        short = ""
        if isinstance(data, dict):
            res = data.get("result")
            if isinstance(res, dict):
                short = str(res.get("shorten_url") or "")
            if not short:
                short = str(data.get("shortenedUrl") or data.get("shortened_url") or "")
        if not short and raw.startswith("http"):
            short = raw
        if not short.startswith("http"):
            raise RuntimeError(f"ShrtFly did not return a link: {raw[:200]}")
        return short

    return retry(call, tries=3, base_delay=3)


def shorten_outputs(outputs, st, vid):
    """Swap each direct download URL for its ShrtFly short link.
    Short links are cached in the state file, so a resumed run never
    creates duplicates."""
    if not SHRTFLY_API_KEY:
        log("  SHRTFLY_API_KEY not set: download buttons will use direct links.")
        return outputs
    cache = st.setdefault("short_urls", {})
    result = []
    for h, dl_url, size, play in outputs:
        short = cache.get(dl_url)
        if not short:
            try:
                short = shorten_url(dl_url)
                cache[dl_url] = short
                save_state(vid, st)
                log(f"  {h}p short link:", short)
            except Exception as ex:  # noqa
                if SHRTFLY_STRICT:
                    raise RuntimeError(f"ShrtFly shortening failed for {h}p: {ex}")
                log(f"  ShrtFly failed for {h}p ({ex}); using direct link.")
                short = dl_url
        result.append((h, short, size, play))
    return result


# ---------- Custom player (plays the Internet Archive MP4 files) ----------
# The player CSS/JS is added once per post. If you paste PLAYER_CSS + PLAYER_JS
# into your Blogger theme once (Theme > Edit HTML), set PLAYER_IN_THEME=true and
# the posts will only contain the small <div class="iap"> block.
PLAYER_IN_THEME = os.environ.get("PLAYER_IN_THEME", "false").lower() == "true"
# Quality that starts first in the player (nearest available is used).
PLAYER_DEFAULT_QUALITY = int(os.environ.get("PLAYER_DEFAULT_QUALITY", "720"))

PLAYER_CSS = r"""<style>
.iap{--u:1px;position:relative;display:block;width:100%;max-width:663px;aspect-ratio:16/9;margin:0 auto;background:#000;overflow:hidden;font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.2;color:#fff;user-select:none;-webkit-user-select:none;-webkit-tap-highlight-color:transparent;text-align:left}
.iap *{box-sizing:border-box}
.iap button{all:unset;box-sizing:border-box;cursor:pointer;color:#fff;font-family:inherit;line-height:1.2;display:flex}
.iap svg{display:block;max-width:none}
.iap-video{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:contain;background:#000;transform-origin:center;transition:transform .2s;margin:0;padding:0;border:0}
.iap .iap-label{position:absolute;top:calc(18*var(--u));left:calc(19*var(--u));z-index:3;background:#34a853;color:#111;font-size:calc(24*var(--u));line-height:1;padding:calc(12*var(--u)) calc(29*var(--u));border-radius:calc(5*var(--u));min-width:calc(198*var(--u));margin:0}
.iap .iap-err{position:absolute;z-index:6;left:calc(12*var(--u));right:calc(12*var(--u));top:50%;transform:translateY(-50%);text-align:center;font-size:calc(18*var(--u));line-height:1.35;background:rgba(0,0,0,.8);padding:calc(14*var(--u));border-radius:calc(8*var(--u));margin:0}
.iap .iap-err[hidden]{display:none}
.iap .iap-big{position:absolute;z-index:3;left:50%;top:50%;width:calc(132*var(--u));height:calc(132*var(--u));margin:calc(-66*var(--u)) 0 0 calc(-66*var(--u));border-radius:50%;background:#34a853;align-items:center;justify-content:center;padding-left:calc(6*var(--u));transition:opacity .2s}
.iap .iap-big svg{width:calc(66*var(--u));height:calc(66*var(--u))}
.iap.playing .iap-big{opacity:0;pointer-events:none}
.iap .iap-bar{position:absolute;z-index:4;left:0;right:0;bottom:0;padding:0 calc(32*var(--u)) calc(19*var(--u)) calc(17*var(--u));transition:opacity .25s;margin:0}
.iap.idle.playing .iap-bar{opacity:0;pointer-events:none}
.iap .iap-seek{position:relative;height:calc(22*var(--u));cursor:pointer;touch-action:none;margin:0 0 calc(12*var(--u)) 0}
.iap .iap-track{position:absolute;left:0;right:0;top:calc(8*var(--u));height:calc(6*var(--u));border-radius:calc(3*var(--u));background:#555;overflow:hidden}
.iap .iap-buf{position:absolute;left:0;top:0;bottom:0;width:0;background:#777}
.iap .iap-prog{position:absolute;left:0;top:0;bottom:0;width:0;background:#34a853}
.iap .iap-thumb{position:absolute;top:0;left:0;width:calc(22*var(--u));height:calc(22*var(--u));border-radius:50%;background:#fff;pointer-events:none}
.iap .iap-row{display:flex;align-items:center;height:calc(56*var(--u));padding:0 calc(4*var(--u))}
.iap .iap-btn{align-items:center;justify-content:center;width:calc(44*var(--u));height:calc(44*var(--u))}
.iap .iap-play{margin-left:calc(20*var(--u))}
.iap .iap-play svg{width:calc(40*var(--u));height:calc(40*var(--u))}
.iap .iap-vol svg{width:calc(36*var(--u));height:calc(36*var(--u))}
.iap .iap-set svg{width:calc(36*var(--u));height:calc(36*var(--u))}
.iap .iap-fs svg{width:calc(34*var(--u));height:calc(34*var(--u))}
.iap .iap-time{font-size:calc(19*var(--u));margin:0 calc(52*var(--u)) 0 calc(32*var(--u));white-space:nowrap}
.iap .iap-gap{flex:1}
.iap .iap-fs{margin-left:calc(36*var(--u))}
.iap .iap-menu{position:absolute;z-index:7;right:calc(19*var(--u));bottom:calc(118*var(--u));width:calc(269*var(--u));background:#fff;color:#111;border-radius:calc(10*var(--u));padding:0;margin:0;font-size:calc(20*var(--u));box-shadow:0 2px 10px rgba(0,0,0,.35);max-height:calc(100% - 134*var(--u));overflow-y:auto;overflow-x:hidden;-webkit-overflow-scrolling:touch;overscroll-behavior:contain;touch-action:pan-y;scrollbar-width:thin}
.iap .iap-menu[hidden]{display:none}
.iap .iap-item{display:flex;align-items:center;justify-content:space-between;width:100%;padding:calc(11*var(--u)) calc(19*var(--u));color:#111;font-size:calc(20*var(--u));white-space:nowrap}
.iap .iap-item:hover{background:#f1f1f1}
.iap .iap-item .v{display:flex;align-items:center;gap:calc(8*var(--u));color:#111}
.iap .iap-item .v svg{width:calc(18*var(--u));height:calc(18*var(--u))}
.iap .iap-item.on .k::before{content:"\2713\00a0\00a0";color:#34a853}
.iap .iap-scale{padding:calc(12*var(--u)) calc(19*var(--u))}
.iap .iap-scale:hover{background:none}
.iap .iap-sb{width:calc(56*var(--u));height:calc(46*var(--u));border-radius:calc(8*var(--u));background:#34a853;color:#fff;font-size:calc(30*var(--u));align-items:center;justify-content:center}
.iap .iap-sv{font-size:calc(22*var(--u));color:#111;cursor:pointer}
.iap .iap-back{font-weight:500;border-bottom:1px solid #e5e5e5;position:sticky;top:0;background:#fff;z-index:1}
.iap .iap-label{transition:opacity .25s}
.iap.idle.playing .iap-label{opacity:0;pointer-events:none}
.iap .iap-spin{display:none;position:absolute;z-index:5;left:50%;top:50%;width:calc(56*var(--u,1px));height:calc(56*var(--u,1px));margin:calc(-28*var(--u,1px)) 0 0 calc(-28*var(--u,1px));border:calc(5*var(--u,1px)) solid rgba(255,255,255,.3);border-top-color:#34a853;border-radius:50%;animation:iapspin .8s linear infinite;pointer-events:none}
.iap.loading .iap-spin{display:block}
@keyframes iapspin{to{transform:rotate(360deg)}}
.iap:fullscreen{max-width:none;aspect-ratio:auto;width:100vw;height:100vh}
</style>"""

PLAYER_JS = r"""<script>
(function(){
  function fixUrl(u){
    u=(u||'').trim().replace(/^http:\/\//i,'https://');
    var m=u.match(/archive\.org\/details\/([^\/?#]+)\/(.+)$/i);
    if(m) u='https://archive.org/download/'+m[1]+'/'+m[2];
    return u;
  }
  document.querySelectorAll('.iap').forEach(function(p){
    if(p.dataset.ready) return; p.dataset.ready=1;
    var $=function(s){return p.querySelector(s)};
    var v=$('.iap-video'), big=$('.iap-big'), menu=$('.iap-menu'), err=$('.iap-err');
    var seek=$('.iap-seek'), prog=$('.iap-prog'), buf=$('.iap-buf'), thumb=$('.iap-thumb');
    var timeEl=$('.iap-time'), pp=$('.iap-pp'), vp=$('.iap-vp');
    var PLAY='M8 5v14l11-7z', PAUSE='M6 19h4V5H6v14zm8-14v14h4V5h-4z';
    var VOL='M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z';
    var MUTE='M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z';

    // Scale the whole UI with the player width
    function rescale(){
      var fs=(document.fullscreenElement===p)||(document.webkitFullscreenElement===p);
      var u=Math.min(p.clientWidth/663,p.clientHeight/373);
      if(fs) u*=0.72;
      u=Math.min(fs?2:1.3,Math.max(fs?0.5:0.4,u));
      p.style.setProperty('--u',u+'px'); updProg();
    }

    if(p.dataset.label) $('.iap-label').textContent=p.dataset.label;
    if(p.dataset.poster) v.poster=p.dataset.poster;

    var sources=[];
    try{ if(p.dataset.sources) sources=JSON.parse(p.dataset.sources); }catch(e){}
    var state={quality:(sources.length?sources[0].label:'Auto'),speed:1,scale:100};
    function showErr(t){ err.textContent=t; err.hidden=false; }
    function loadSrc(src,keepTime,wasPlaying){
      err.hidden=true; src=fixUrl(src);
      if(!src){ showErr('Video link missing.'); return; }
      var t=keepTime?v.currentTime:0; v.src=src; v.load();
      v.addEventListener('loadedmetadata',function f(){v.removeEventListener('loadedmetadata',f); if(t) v.currentTime=t; v.playbackRate=state.speed; if(wasPlaying) v.play();});
    }
    v.addEventListener('error',function(){
      var c=v.error?v.error.code:0;
      showErr(c===4 ? 'This video format is not supported by your browser.' : 'Video could not be loaded. Check your internet and try again.');
    });
    loadSrc(sources.length?sources[0].src:p.dataset.src,false,false);

    function fmt(s){ if(!isFinite(s)) s=0; s=Math.floor(s); var h=Math.floor(s/3600), m=Math.floor(s%3600/60), x=s%60;
      return (h?h+':'+(m<10?'0':''):'')+m+':'+(x<10?'0':'')+x; }
    var totalDur=parseFloat(p.dataset.duration)||0;
    function curDur(){ return (isFinite(v.duration)&&v.duration>0)?v.duration:totalDur; }
    function updTime(){ timeEl.innerHTML=fmt(v.currentTime)+'&nbsp; - &nbsp;'+fmt(curDur()); }
    function updProg(){
      var d=curDur(), pct=d?v.currentTime/d:0, w=seek.clientWidth-thumb.offsetWidth;
      thumb.style.transform='translateX('+(pct*w)+'px)';
      prog.style.width=(pct*100)+'%'; updTime();
    }
    function toggle(){
      if(v.paused){ var pr=v.play(); if(pr&&pr.catch) pr.catch(function(){ if(!err.hidden) return; showErr('Cannot play this video.'); }); }
      else v.pause();
    }

    big.onclick=toggle; $('.iap-play').onclick=toggle;
    v.onclick=function(){ if(menu.hidden) toggle(); else menu.hidden=true; };
    v.addEventListener('play',function(){p.classList.add('playing'); pp.setAttribute('d',PAUSE); poke();});
    v.addEventListener('pause',function(){p.classList.remove('playing'); pp.setAttribute('d',PLAY);});
    v.addEventListener('ended',function(){p.classList.remove('playing'); pp.setAttribute('d',PLAY);});
    v.addEventListener('timeupdate',updProg);
    function spinOn(){ p.classList.add('loading'); }
    function spinOff(){ p.classList.remove('loading'); }
    ['waiting','seeking','loadstart'].forEach(function(n){ v.addEventListener(n,function(){ if(!v.paused||n!=='loadstart') spinOn(); }); });
    v.addEventListener('play',function(){ if(v.readyState<3) spinOn(); });
    ['playing','canplay','canplaythrough','seeked','pause','ended','error'].forEach(function(n){ v.addEventListener(n,spinOff); });
    v.addEventListener('loadedmetadata',updProg);
    v.addEventListener('progress',function(){ try{ if(v.buffered.length) buf.style.width=(v.buffered.end(v.buffered.length-1)/v.duration*100)+'%'; }catch(e){} });
    window.addEventListener('resize',rescale);
    document.addEventListener('fullscreenchange',function(){setTimeout(rescale,50)});
    if(window.ResizeObserver) new ResizeObserver(rescale).observe(p);

    var dragging=false;
    function seekTo(e){
      var r=seek.getBoundingClientRect();
      var pct=Math.min(1,Math.max(0,(e.clientX-r.left-thumb.offsetWidth/2)/(r.width-thumb.offsetWidth)));
      if(v.duration) v.currentTime=pct*v.duration; updProg();
    }
    seek.addEventListener('pointerdown',function(e){dragging=true; seek.setPointerCapture(e.pointerId); seekTo(e);});
    seek.addEventListener('pointermove',function(e){ if(dragging) seekTo(e); });
    seek.addEventListener('pointerup',function(){dragging=false;});

    $('.iap-vol').onclick=function(){ v.muted=!v.muted; vp.setAttribute('d',v.muted?MUTE:VOL); };

    $('.iap-fs').onclick=function(){
      if(document.fullscreenElement||document.webkitFullscreenElement){ (document.exitFullscreen||document.webkitExitFullscreen).call(document); }
      else { var f=p.requestFullscreen||p.webkitRequestFullscreen; if(f) f.call(p); else if(v.webkitEnterFullscreen) v.webkitEnterFullscreen(); }
    };

    var timer; function poke(){ p.classList.remove('idle'); clearTimeout(timer); timer=setTimeout(function(){ if(menu.hidden) p.classList.add('idle'); },2800); }
    p.addEventListener('mousemove',poke); p.addEventListener('touchstart',poke,{passive:true});

    var CH='<svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6" fill="none" stroke="#111" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    var speeds=[[0.5,'0.5x'],[0.75,'0.75x'],[1,'Normal'],[1.25,'1.25x'],[1.5,'1.5x'],[2,'2x']];
    function speedName(){ return speeds.filter(function(s){return s[0]===state.speed})[0][1]; }
    function main(){
      menu.innerHTML=''; menu.scrollTop=0;
      [['Quality',state.quality,qualityMenu],['Speed',speedName(),speedMenu],['Scale',state.scale+'%',scaleMenu]].forEach(function(r){
        var b=document.createElement('button'); b.className='iap-item';
        b.innerHTML='<span class="k">'+r[0]+'</span><span class="v">'+r[1]+CH+'</span>';
        b.onclick=function(e){e.stopPropagation(); r[2]();}; menu.appendChild(b);
      });
    }
    function sub(title,list,cur,pick){
      menu.innerHTML='';
      var h=document.createElement('button'); h.className='iap-item iap-back';
      h.innerHTML='<span class="k">&#8249;&nbsp; '+title+'</span>'; h.onclick=function(e){e.stopPropagation(); main();}; menu.appendChild(h);
      list.forEach(function(it){
        var b=document.createElement('button'); b.className='iap-item'+(it[0]===cur?' on':'');
        b.innerHTML='<span class="k">'+it[1]+'</span>';
        b.onclick=function(e){e.stopPropagation(); pick(it[0]); main();}; menu.appendChild(b);
      });
      var on=menu.querySelector('.on');
      menu.scrollTop=on?Math.max(0,on.offsetTop-menu.clientHeight/2+on.offsetHeight/2):0;
    }
    function qualityMenu(){
      var list=sources.map(function(s){return [s.label,s.label];}).sort(function(a,b){return parseInt(b[0])-parseInt(a[0]);});
      sub('Quality',list,state.quality,function(q){
        state.quality=q; var s=sources.filter(function(x){return x.label===q})[0]||sources[0];
        if(s) loadSrc(s.src,true,!v.paused);
      });
    }
    function speedMenu(){ sub('Speed',speeds,state.speed,function(s){state.speed=s; v.playbackRate=s;}); }
    function scaleMenu(){
      menu.innerHTML='';
      var h=document.createElement('button'); h.className='iap-item iap-back';
      h.innerHTML='<span class="k">&#8249;&nbsp; Scale</span>'; h.onclick=function(e){e.stopPropagation(); main();}; menu.appendChild(h);
      var row=document.createElement('div'); row.className='iap-item iap-scale';
      row.innerHTML='<button class="iap-sb" aria-label="Zoom out">&minus;</button><span class="iap-sv">'+state.scale+'%</span><button class="iap-sb" aria-label="Zoom in">+</button>';
      menu.appendChild(row);
      var sv=row.querySelector('.iap-sv'), bs=row.querySelectorAll('.iap-sb');
      function set(n){ state.scale=Math.min(200,Math.max(50,n)); v.style.transform='scale('+(state.scale/100)+')'; sv.textContent=state.scale+'%'; }
      bs[0].onclick=function(e){e.stopPropagation(); set(state.scale-10);};
      bs[1].onclick=function(e){e.stopPropagation(); set(state.scale+10);};
      sv.onclick=function(e){e.stopPropagation(); set(100);};
    }

    $('.iap-set').onclick=function(e){ e.stopPropagation(); if(menu.hidden){ main(); menu.hidden=false; } else menu.hidden=true; };
    document.addEventListener('click',function(e){ if(!menu.hidden && !menu.contains(e.target)) menu.hidden=true; });

    p.tabIndex=0;
    p.addEventListener('keydown',function(e){
      if(e.code==='Space'){e.preventDefault(); toggle();}
      if(e.code==='ArrowRight') v.currentTime+=5;
      if(e.code==='ArrowLeft') v.currentTime-=5;
    });
    rescale();
  });
})();
</script>"""

PLAYER_DIV = r"""<div class="iap" data-src="__SRC__" data-sources="__SOURCES__" data-label="__LABEL__" data-poster="__POSTER__" data-duration="__DURATION__">
  <video class="iap-video" playsinline preload="metadata"></video>
  <div class="iap-label">__LABEL_TEXT__</div>
  <div class="iap-err" hidden></div>
  <div class="iap-spin"></div>
  <button class="iap-big" aria-label="Play">
    <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" fill="#000"/></svg>
  </button>
  <div class="iap-menu" hidden></div>
  <div class="iap-bar">
    <div class="iap-seek">
      <div class="iap-track"><div class="iap-buf"></div><div class="iap-prog"></div></div>
      <div class="iap-thumb"></div>
    </div>
    <div class="iap-row">
      <button class="iap-btn iap-play" aria-label="Play/Pause">
        <svg viewBox="0 0 24 24"><path class="iap-pp" d="M8 5v14l11-7z" fill="#fff"/></svg>
      </button>
      <span class="iap-time">0:00&nbsp; - &nbsp;__DURATION_TEXT__</span>
      <button class="iap-btn iap-vol" aria-label="Mute">
        <svg viewBox="0 0 24 24"><path class="iap-vp" d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" fill="#fff"/></svg>
      </button>
      <span class="iap-gap"></span>
      <button class="iap-btn iap-set" aria-label="Settings">
        <svg viewBox="0 0 24 24"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z" fill="#fff"/></svg>
      </button>
      <button class="iap-btn iap-fs" aria-label="Fullscreen">
        <svg viewBox="0 0 24 24"><path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z" fill="#fff"/></svg>
      </button>
    </div>
  </div>
</div>"""


def build_player(outputs, label, poster="", duration=0):
    """Custom player block. outputs = [(height, download_url, size, play_mp4_url), ...].
    The Quality menu lists every converted file; PLAYER_DEFAULT_QUALITY starts first."""
    e = html.escape
    items = sorted(outputs, key=lambda o: abs(int(o[0]) - PLAYER_DEFAULT_QUALITY))
    first = items[0][3]
    # default first, then the rest from high to low quality in the menu
    rest = sorted(items[1:], key=lambda o: -int(o[0]))
    sources = [{"label": f"{int(o[0])}p", "src": o[3]} for o in [items[0]] + rest]
    label = str(label or "").strip() or "Watch Online"
    div = (PLAYER_DIV
           .replace("__SRC__", e(first, quote=True))
           .replace("__SOURCES__", e(json.dumps(sources), quote=True))
           .replace("__LABEL__", e(label, quote=True))
           .replace("__LABEL_TEXT__", e(label))
           .replace("__POSTER__", e(poster or "", quote=True))
           .replace("__DURATION__", str(int(duration or 0)))
           .replace("__DURATION_TEXT__", fmt_clock(duration)))
    if PLAYER_IN_THEME:
        return div
    return PLAYER_CSS + "\n" + div + "\n" + PLAYER_JS


# ---------- MP4 (watch) + MKV (download) helpers ----------
def remux_copy(src_mp4, out_path):
    """Repack the finished MP4 into another container (e.g. mkv) WITHOUT re-encoding.
    Takes seconds, quality stays identical."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_mp4,
         "-map", "0:v:0", "-map", "0:a?", "-c", "copy", out_path])


def fetch_file(url, dest):
    """Download a file from Internet Archive (used only when resuming after a restart)."""
    def go():
        urllib.request.urlretrieve(url, dest)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise RuntimeError("downloaded file is empty")
    retry(go, tries=3, base_delay=20)


# ---------- ffmpeg helpers ----------
def parse_fps(*vals):
    for v in vals:
        try:
            a, b = str(v).split("/")
            a, b = float(a), float(b)
            if b and a / b > 0:
                return a / b
        except Exception:  # noqa
            pass
    return 30.0


def fmt_fps(x):
    r = round(x)
    return str(r) if abs(x - r) < 0.05 else f"{x:.2f}"


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", path])
    d = json.loads(out)
    st = d["streams"][0]
    fps = parse_fps(st.get("avg_frame_rate"), st.get("r_frame_rate"))
    return float(d["format"]["duration"]), int(st["width"]), int(st["height"]), fps


def webp_args(quality, lossless=False):
    """Shared high-quality libwebp encoder arguments for ffmpeg."""
    args = [
        "-c:v", "libwebp",
        "-quality", str(quality),
        "-compression_level", str(WEBP_COMPRESSION),
        "-preset", "picture",
        "-pix_fmt", "yuv420p",
    ]
    if lossless:
        args += ["-lossless", "1"]
    return args


# ---------- AI Scene Intelligence screenshots ----------
def _scene_timestamps(src, dur):
    candidates = []
    vf = "fps=2,select='gt(scene,0.28)',showinfo"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "info", "-i", src,
             "-vf", vf, "-an", "-f", "null", "-"],
            capture_output=True, text=True, check=True
        )
        raw = (proc.stderr or "") + (proc.stdout or "")
        for m in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", raw):
            t = float(m.group(1))
            if 2.0 < t < max(2.0, dur - 2.0):
                candidates.append(t)
    except Exception as ex:
        log("  Scene detection failed:", ex)

    candidates = sorted(set(round(x, 2) for x in candidates))
    spaced = []
    min_gap = max(4.0, dur / 80.0)
    for t in candidates:
        if not spaced or t - spaced[-1] >= min_gap:
            spaced.append(t)

    fallback = [
        dur * i / 16.0
        for i in range(1, 16)
        if 2.0 < dur * i / 16.0 < dur - 2.0
    ]
    merged = sorted(set(spaced + [round(x, 2) for x in fallback]))

    if len(merged) > 36:
        selected = []
        step = (len(merged) - 1) / 35
        for i in range(36):
            selected.append(merged[round(i * step)])
        merged = sorted(set(selected))

    log(f"  Scene Intelligence: {len(merged)} candidate timestamps")
    return merged


def _frame_brightness(src, t):
    """Average brightness (0-255) of the frame at time t."""
    try:
        raw = subprocess.check_output([
            "ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", src,
            "-frames:v", "1", "-vf", "scale=32:18,format=gray",
            "-f", "rawvideo", "-"])
        return sum(raw) / len(raw) if raw else 0
    except Exception:  # noqa
        return 255


def make_player_poster(src, dur, outdir):
    """Player thumbnail = the FIRST frame of the video (not a screenshot).
    If the very first frame is pure black (common fade-in), use the first
    non-black frame within the first seconds. Full frame, no crop, so it
    matches exactly what the player shows."""
    t = 0.0
    for cand in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        if cand >= dur:
            break
        t = cand
        if _frame_brightness(src, cand) > 12:
            break
    out = str(Path(outdir) / "player_poster.webp")
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1",
        "-vf", "scale='min(1280,iw)':-2:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1",
    ] + webp_args(SHOT_WEBP_QUALITY) + [out])
    log(f"  Player poster: first frame at {t:.1f}s")
    return out


def _detect_crop(src, dur, w, h):
    crops = []
    for t in [dur * 0.20, dur * 0.40, dur * 0.60, dur * 0.80]:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "info",
                 "-ss", f"{t:.3f}", "-i", src,
                 "-frames:v", "1",
                 "-vf", "cropdetect=limit=24:round=2:reset=0",
                 "-f", "null", "-"],
                capture_output=True, text=True, check=True
            )
            raw = (proc.stderr or "") + (proc.stdout or "")
            matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", raw)
            if matches:
                cw, ch, cx, cy = map(int, matches[-1])
                if cw >= int(w * 0.70) and ch >= int(h * 0.70):
                    crops.append((cw, ch, cx, cy))
        except Exception as ex:
            log("  cropdetect sample skipped:", ex)

    if not crops:
        return w, h, 0, 0

    from collections import Counter
    best, count = Counter(crops).most_common(1)[0]

    if count >= 2:
        log(f"  Black-bar detection: crop={best[0]}:{best[1]}:{best[2]}:{best[3]} "
            f"({count}/{len(crops)} samples)")
        return best

    cw, ch, cx, cy = crops[0]
    if cw < w * 0.98 or ch < h * 0.98:
        log(f"  Black-bar detection: crop={cw}:{ch}:{cx}:{cy}")
        return crops[0]

    return w, h, 0, 0


def _make_scene_candidates(src, timestamps, base_crop, outdir):
    candidate_dir = outdir / "scene_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    bw, bh, bx, by = base_crop
    ratio = bw / bh if bh else 16 / 9

    if ratio > 16 / 9:
        pw = int(bh * 16 / 9)
        ph = bh
        px = bx + (bw - pw) // 2
        py = by
    elif ratio < 16 / 9:
        pw = bw
        ph = int(bw * 9 / 16)
        px = bx
        py = by + (bh - ph) // 2
    else:
        pw, ph, px, py = bw, bh, bx, by

    # Keep crop strictly inside the source frame.
    pw = min(pw, w := max(2, bw))
    ph = min(ph, h := max(2, bh))
    px = max(bx, min(int(px), bx + bw - pw))
    py = max(by, min(int(py), by + bh - ph))

    pw = max(2, (pw // 2) * 2)
    ph = max(2, (ph // 2) * 2)

    files = []
    for i, t in enumerate(timestamps):
        p = candidate_dir / f"candidate_{i:02d}.jpg"
        vf = f"crop={pw}:{ph}:{px}:{py},scale=768:432:flags=lanczos,setsar=1"
        try:
            run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{t:.3f}",
                "-i", src,
                "-frames:v", "1",
                "-vf", vf,
                "-q:v", "3",
                "-pix_fmt", "yuvj420p",
                str(p),
            ])
            files.append((t, p))
        except Exception as ex:
            log(f"  Candidate frame {i} failed:", ex)

    return files, (pw, ph, px, py)


def _gemini_scene_select_rest(model, prompt, candidates):
    """Call Gemini Scene Intelligence through REST, avoiding SDK AFC warnings."""
    parts = [{"text": prompt}]
    for idx, (_, p) in enumerate(candidates):
        parts.append({"text": f"Candidate index: {idx}"})
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(p.read_bytes()).decode("ascii"),
            }
        })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(model, safe="")
        + ":generateContent?key="
        + urllib.parse.quote(GEMINI_KEY, safe="")
    )
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "MovieBot/1.0",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _ensure_local_qwen():
    """Return the local llama command, installing llama.cpp only if needed."""
    if not LOCAL_AI_ENABLED:
        raise RuntimeError("Local AI fallback is disabled (LOCAL_AI_ENABLED=false).")

    for cmd in ("llama", "llama-cli"):
        if shutil.which(cmd):
            return cmd

    log("  Local AI: llama.cpp not found; installing the official llama binary...")
    install_script = "https://llama.app/install.sh"
    proc = subprocess.run(
        ["bash", "-lc", f"curl -LsSf {shlex.quote(install_script)} | sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama.cpp installation failed: {proc.stdout[-2000:]}")

    local_bin = str(Path.home() / ".local" / "bin")
    os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")
    for cmd in ("llama", "llama-cli"):
        if shutil.which(cmd):
            return cmd

    raise RuntimeError("llama.cpp installed but no llama executable was found.")


def _local_qwen_json(prompt, image_paths=None, max_tokens=700):
    """Run Qwen3-VL locally and parse its JSON response."""
    cmd = _ensure_local_qwen()
    image_paths = [str(x) for x in (image_paths or [])][:LOCAL_AI_MAX_IMAGES]
    if not image_paths:
        raise RuntimeError("Qwen fallback requires at least one image.")

    image_arg = ",".join(image_paths)
    if cmd == "llama":
        executable = [cmd, "cli"]
    else:
        executable = [cmd]

    args = executable + [
        "-hf", LOCAL_QWEN_MODEL,
        "--image", image_arg,
        "-p", prompt,
        "-c", str(LOCAL_AI_CONTEXT),
        "-n", str(max_tokens),
        "-t", str(LOCAL_AI_THREADS),
        "--temperature", "0.1",
        "--reasoning", "off",
        "--simple-io",
        "--single-turn",
        "--no-warmup",
    ]

    log("  Local AI: running Qwen3-VL fallback...")
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=LOCAL_AI_TIMEOUT,
    )
    raw = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(f"Qwen3-VL exited with code {proc.returncode}: {raw[-2500:]}")

    # llama.cpp may print status/timing lines around the model response.
    candidates = re.findall(r"\{.*\}", raw, flags=re.S)
    for candidate in reversed(candidates):
        cleaned = candidate.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as ex:
        raise RuntimeError(f"Qwen3-VL returned invalid JSON: {raw[-3000:]}") from ex


def _local_qwen_scene_select(candidates, count):
    """Select screenshot indexes with Qwen3-VL after Gemini scene selection fails."""
    if not candidates:
        return []
    wanted = min(count, len(candidates))
    prompt = f"""You are selecting screenshots for an original film website.
Choose exactly {wanted} candidate frame indexes.
Prefer sharp, cinematic, well-lit frames with people, emotion, action, scenery, or strong composition.
Avoid black frames, blur, credits, logos, title cards, empty frames, and near-duplicates.
Spread choices across the movie when possible.
Return ONLY JSON: {{\"selected\":[0,1,2]}}"""
    paths = [p for _, p in candidates]
    data = _local_qwen_json(prompt, paths, max_tokens=200)
    selected = []
    for value in data.get("selected") or []:
        try:
            idx = int(value)
            if 0 <= idx < len(candidates):
                selected.append(idx)
        except (TypeError, ValueError):
            pass
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise RuntimeError("Qwen3-VL returned no valid screenshot indexes.")
    return selected[:wanted]


def _local_qwen_analyze(filename_hint, frame_paths, site_labels):
    """Metadata fallback using local Qwen3-VL."""
    hint = clean_hint(filename_hint)
    year = find_year(filename_hint)
    prompt = f"""You are the backup metadata editor for an ORIGINAL movie blog.
Gemini failed, so you must complete the metadata task locally.
Filename hint: {hint}
Language hint: {LANGUAGE_HINT}
Director: {DIRECTOR_NAME}
Release year from filename: {year or 'Unknown'}
Manual title supplied by the site owner: {MANUAL_TITLE or 'None'}

Use the supplied movie frames, audio context, filename hint, and the manual title when present.
When a manual title is supplied, treat that title as the authoritative movie name and do NOT try to replace it.
Identify the release year when it can reasonably be established from the supplied video/context; otherwise return null.
Do not invent cast, crew, awards, ratings, box office, exact plot facts, or IMDb information.
For the title, first read any visible movie title/title card and use it; if the filename clearly contains
the real title, clean it and preserve it. Never use an actor name, character name, genre or generic phrase
as the title when a real title is visible. Invent a title only when no real title can reasonably be identified.
Return ONLY valid JSON with:
title: clean 1-8 word film title, preferably the actual visible/filename title
tagline: one short sentence, max 18 words
release_year: 4-digit release year when reasonably identifiable, otherwise null
description: 2 compact paragraphs, about 120-180 words total, written as a proper full-movie synopsis even when the supplied video is only a short clip; do not describe only the clip scene
language: display language(s) if reasonably identifiable, otherwise the language hint or Unknown
original_language: main/original language if reasonably identifiable, otherwise Unknown
genres: 1-3 genres
content_rating: General audience, Teen and above, or Mature audience
tags: up to 6 short keywords
labels: pick 1-4 values ONLY from this exact list: {json.dumps(site_labels)}
Do not use piracy terms such as leaked, HD print, free download full movie, WEB-DL, dual audio, or 300mb."""
    return _local_qwen_json(prompt, frame_paths, max_tokens=700)


def _ai_choose_scene_frames(candidates, count):
    if not candidates:
        return []

    candidates = candidates[:36]
    wanted = min(count, len(candidates))

    prompt = f"""You are selecting screenshots for an ORIGINAL film website.

Choose exactly {wanted} of the supplied candidate frames.

Selection rules:
- Prefer sharp, cinematic, visually interesting and well-lit frames.
- Prefer strong characters, environments, action, emotion, or composition.
- Avoid black frames, blurry frames, transitional frames, credits, logos,
  title cards, empty shots, and frames dominated by darkness.
- Avoid near-duplicate frames.
- Spread selections across different parts of the movie when possible.
- Do not infer or invent plot facts.

Return ONLY valid JSON:
{{"selected":[0,1,2]}}
"""

    # Scene selection does not need function calling or tools. Use the REST
    # generateContent endpoint directly so the SDK's automatic-function-calling
    # path is not involved. A temporary 503 should not kill the workflow.
    scene_models = []
    for model in [
        os.environ.get("SCENE_GEMINI_MODEL", "gemini-3.5-flash"),
        GEMINI_MODEL,
        "gemini-3.5-flash",
    ]:
        if model and model not in scene_models:
            scene_models.append(model)

    for model in scene_models:
        for attempt in range(1, 4):
            try:
                response = _gemini_scene_select_rest(model, prompt, candidates)
                texts = []
                for candidate in response.get("candidates") or []:
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        if part.get("text"):
                            texts.append(part["text"])
                raw_text = "\n".join(texts).strip()
                if not raw_text:
                    raise RuntimeError("Gemini returned no scene-selection text.")

                cleaned = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    raw_text,
                    flags=re.I,
                ).strip()
                data = json.loads(cleaned)

                selected = []
                for value in data.get("selected") or []:
                    try:
                        idx = int(value)
                        if 0 <= idx < len(candidates):
                            selected.append(idx)
                    except (TypeError, ValueError):
                        pass

                selected = list(dict.fromkeys(selected))
                if selected:
                    log(f"  AI selected scene frames ({model}): {selected[:wanted]}")
                    return selected[:wanted]

                raise RuntimeError("Gemini returned an empty/invalid selected list.")

            except urllib.error.HTTPError as ex:
                detail = ""
                try:
                    detail = ex.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                if ex.code in (408, 429, 500, 502, 503, 504):
                    if attempt < 3:
                        wait = 2 ** attempt
                        log(
                            f"  Scene selection {model}: HTTP {ex.code}; "
                            f"retrying in {wait}s ({attempt}/3)"
                        )
                        time.sleep(wait)
                        continue
                    log(f"  Scene selection {model}: HTTP {ex.code} after 3 attempts: {detail}")
                else:
                    log(f"  Scene selection {model}: HTTP {ex.code}: {detail}")
                break

            except (urllib.error.URLError, TimeoutError) as ex:
                if attempt < 3:
                    wait = 2 ** attempt
                    log(
                        f"  Scene selection {model}: temporary network error; "
                        f"retrying in {wait}s ({attempt}/3): {ex}"
                    )
                    time.sleep(wait)
                    continue
                log(f"  Scene selection {model}: network error after 3 attempts: {ex}")
                break

            except Exception as ex:
                log(f"  Gemini scene selection failed ({model}): {ex}")
                break

    # Gemini failed completely for this task: use local Qwen3-VL before the
    # deterministic selector. This path is never reached when Gemini succeeds.
    try:
        selected = _local_qwen_scene_select(candidates, wanted)
        log(f"  AI selected scene frames (local Qwen3-VL): {selected}")
        return selected
    except Exception as ex:
        log(f"  Local Qwen3-VL scene selection failed: {ex}")

    # Deterministic fallback if both Gemini and local AI are unavailable.
    # The screenshot pipeline therefore continues even during a Gemini outage.
    log("  Scene Intelligence: Gemini unavailable; using deterministic frame selection.")
    if len(candidates) <= wanted:
        return list(range(len(candidates)))

    step = (len(candidates) - 1) / max(1, wanted - 1)
    return [round(i * step) for i in range(wanted)]


def make_screenshots(src, dur, w, h, outdir):
    """
    AI Scene Intelligence screenshot pipeline.

    Final screenshots:
      - exact 16:9
      - encoded black letterbox bars removed where detectable
      - no artificial borders
      - up to 1920x1080
      - high-quality WebP (shot_N.webp) for upload
      - a lossless-quality JPG twin (shot_N.jpg) kept only for AI analysis
      - selected by Gemini from scene candidates
      - extracted from the original source
    """
    base_crop = _detect_crop(src, dur, w, h)
    timestamps = _scene_timestamps(src, dur)

    minimum_candidates = max(8, SCREENSHOTS * 2)
    if len(timestamps) < minimum_candidates:
        extra = [
            dur * i / (minimum_candidates + 1)
            for i in range(1, minimum_candidates + 1)
            if 2.0 < dur * i / (minimum_candidates + 1) < dur - 2.0
        ]
        timestamps = sorted(set(
            timestamps + [round(x, 2) for x in extra]
        ))

    candidates, crop16 = _make_scene_candidates(
        src, timestamps, base_crop, outdir
    )

    chosen_indexes = _ai_choose_scene_frames(candidates, SCREENSHOTS)

    if not chosen_indexes:
        chosen_indexes = list(range(min(SCREENSHOTS, len(candidates))))

    # Fill missing slots without duplicating a selected candidate.
    used = set(chosen_indexes)
    for i in range(len(candidates)):
        if len(chosen_indexes) >= SCREENSHOTS:
            break
        if i not in used:
            chosen_indexes.append(i)
            used.add(i)

    bw, bh, bx, by = crop16

    # crop16 is already 16:9. Never add padding or black bars.
    if bw >= 1920:
        out_w, out_h = 1920, 1080
    elif bw >= 1280:
        out_w, out_h = 1280, 720
    else:
        out_w = max(2, (bw // 2) * 2)
        out_h = max(2, (int(out_w * 9 / 16) // 2) * 2)

    files = []
    for out_index, candidate_index in enumerate(
        chosen_indexes[:SCREENSHOTS], 1
    ):
        t = candidates[candidate_index][0]
        jpg_p = str(outdir / f"shot_{out_index}.jpg")
        webp_p = str(outdir / f"shot_{out_index}.webp")

        vf = (
            f"crop={bw}:{bh}:{bx}:{by},"
            f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_int,"
            "setsar=1"
        )

        # High-quality JPG: used ONLY as input for Gemini/Qwen analysis.
        run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", src,
            "-frames:v", "1",
            "-vf", vf,
            "-q:v", "1",
            "-pix_fmt", "yuvj420p",
            jpg_p,
        ])

        # Final WebP: this is the file that gets uploaded and shown on the post.
        run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", src,
            "-frames:v", "1",
            "-vf", vf,
        ] + webp_args(SHOT_WEBP_QUALITY) + [webp_p])

        files.append(webp_p)
        log(
            f"  Screenshot {out_index}: {t:.2f}s -> "
            f"{out_w}x{out_h}, AI-selected, WebP q{SHOT_WEBP_QUALITY}"
        )

    return files


def _gemini_google_image_search(query, outdir):
    """
    Use Gemini's Google Search grounding with image_search enabled to find
    a real web image for the movie poster/thumbnail.
    Returns a list of candidate image URLs.
    """
    prompt = f"""Search Google Images for the movie/film poster for:
"{query}"

Find the most relevant official or professionally published poster/cover.
Prefer a clean portrait movie poster, ideally close to 2:3 ratio.
Avoid fan edits, screenshots, social-media collages, unrelated films,
logos-only images, and images with large watermarks.

Use image search. Return only a short JSON object:
{{"query":"{query}"}}.
Do not invent image URLs; the application will read the image-search results.
"""

    candidates = []

    for model in dict.fromkeys([GEMINI_MODEL, "gemini-flash-latest"]):
        try:
            payload = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "tools": [
                    {
                        "google_search": {
                            "search_types": {
                                "image_search": {}
                            }
                        }
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.1,
                },
            }

            endpoint = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + urllib.parse.quote(model, safe="")
                + ":generateContent?key="
                + urllib.parse.quote(GEMINI_KEY, safe="")
            )

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "MovieBot/1.0",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))

            # REST response uses camelCase. Accept snake_case too so the
            # function remains tolerant of SDK/proxy transformations.
            candidates_json = data.get("candidates") or []
            for candidate in candidates_json:
                gm = (
                    candidate.get("groundingMetadata")
                    or candidate.get("grounding_metadata")
                    or {}
                )
                chunks = (
                    gm.get("groundingChunks")
                    or gm.get("grounding_chunks")
                    or []
                )

                for chunk in chunks:
                    image = chunk.get("image") or {}
                    image_uri = (
                        image.get("imageUri")
                        or image.get("image_uri")
                    )
                    if image_uri:
                        candidates.append(image_uri)

                # If the model happened to return a URL in text, keep it as
                # a secondary candidate; grounding image URLs remain preferred.
                for part in (candidate.get("content") or {}).get("parts") or []:
                    value = part.get("text") or ""
                    for url in re.findall(r"https?://[^\s\"'<>]+", value):
                        candidates.append(url.rstrip(".,)"))

            candidates = list(dict.fromkeys(candidates))
            if candidates:
                log(f"  Google Image Search found {len(candidates)} image candidate(s)")
                return candidates[:8]

            log(f"  Gemini Google Image Search returned no image candidates ({model})")

        except Exception as ex:
            log(f"  Gemini Google Image Search failed ({model}): {ex}")

    return []


def _download_web_image(url, path):
    """Download one image-search result without loading an entire movie."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        },
    )

    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()

    if len(data) < 2048:
        raise RuntimeError("Downloaded image is too small.")

    Path(path).write_bytes(data)


def _make_2x3_thumbnail_from_image(src_image, out_path):
    """
    Convert a web poster to a clean 2:3 portrait high-quality WebP thumbnail.
    No 9:16 crop and no artificial black borders.
    """
    probe_json = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", str(src_image)
    ])
    info = json.loads(probe_json)
    stream = info["streams"][0]
    iw = int(stream["width"])
    ih = int(stream["height"])

    if iw <= 0 or ih <= 0:
        raise RuntimeError("Invalid downloaded image dimensions.")

    # Target poster ratio = 2:3, matching the reference-style movie cards.
    if iw / ih > 2 / 3:
        ch = ih
        cw = int(ih * 2 / 3)
        cx = (iw - cw) // 2
        cy = 0
    else:
        cw = iw
        ch = int(iw * 3 / 2)
        cx = 0
        cy = (ih - ch) // 2

    cw = max(2, (cw // 2) * 2)
    ch = max(2, (ch // 2) * 2)
    cx = max(0, min(cx, iw - cw))
    cy = max(0, min(cy, ih - ch))

    # THUMB_W x THUMB_H = exact 2:3 (default 1000x1500). NOT 9:16.
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_image),
        "-vf",
        f"crop={cw}:{ch}:{cx}:{cy},"
        f"scale={THUMB_W}:{THUMB_H}:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1",
        "-frames:v", "1",
    ] + webp_args(THUMB_WEBP_QUALITY) + [str(out_path)])

    return str(out_path)


def make_thumbnail(src, dur, w, h, outdir, movie_title="", filename_hint="",
                   poster_urls=None):
    """
    Poster thumbnail (high-quality WebP, always exact 2:3).

    Priority:
      1. Official poster(s) from TMDB / OMDb (IMDb)   <- poster_urls
      2. Gemini + Google Image Search result
      3. 2:3 crop from the movie source frame (last fallback)
    """
    title = str(movie_title or "").strip()
    hint = clean_hint(filename_hint) if filename_hint else ""
    query = title or hint or "movie poster"
    if hint and title and hint.lower() not in title.lower():
        query = f"{title} {hint}"

    raw_dir = outdir / "web_thumbnail_candidates"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) TMDB / OMDb official posters ----
    poster_urls = list(poster_urls or [])
    for i, url in enumerate(poster_urls, 1):
        raw_path = raw_dir / f"official_{i}.source"
        final_path = outdir / "thumb_2x3.webp"
        try:
            log(f"  Trying TMDB/IMDb poster {i}/{len(poster_urls)}")
            _download_web_image(url, raw_path)
            _make_2x3_thumbnail_from_image(raw_path, final_path)
            if final_path.exists() and final_path.stat().st_size > 10_000:
                log("  Thumbnail source: TMDB/OMDb official poster")
                log(f"  Thumbnail size: {THUMB_W}x{THUMB_H} (2:3) WebP q{THUMB_WEBP_QUALITY}")
                return str(final_path)
        except Exception as ex:
            log(f"  Official poster {i} failed: {ex}")
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ---- 2) Gemini + Google Image Search ----
    image_urls = _gemini_google_image_search(
        f"{query} official movie poster",
        raw_dir
    )

    for i, image_url in enumerate(image_urls, 1):
        raw_path = raw_dir / f"poster_{i}.source"
        final_path = outdir / "thumb_2x3.webp"

        try:
            log(f"  Trying Google image poster {i}/{len(image_urls)}")
            _download_web_image(image_url, raw_path)
            _make_2x3_thumbnail_from_image(raw_path, final_path)

            if final_path.exists() and final_path.stat().st_size > 10_000:
                log("  Thumbnail source: Google Image Search")
                log(f"  Thumbnail size: {THUMB_W}x{THUMB_H} (2:3) WebP q{THUMB_WEBP_QUALITY}")
                return str(final_path)

        except Exception as ex:
            log(f"  Google poster candidate {i} failed: {ex}")
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ---- 3) Safe fallback: source-frame 2:3 ----
    log("  Official/Google poster unavailable; using source-frame 2:3 fallback.")

    if w / h > 2 / 3:
        ch = h
        cw = int(h * 2 / 3)
    else:
        cw = w
        ch = int(w * 3 / 2)

    cw = max(2, (cw // 2) * 2)
    ch = max(2, (ch // 2) * 2)

    p = str(outdir / "thumb_2x3.webp")
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{dur * 0.35:.2f}",
        "-i", src,
        "-frames:v", "1",
        "-vf",
        f"crop={cw}:{ch},scale={THUMB_W}:{THUMB_H}:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1",
    ] + webp_args(THUMB_WEBP_QUALITY) + [p])
    return p


def analysis_inputs(src, dur, outdir, screenshot_paths):
    """Reuse the final screenshots for Gemini; only extract the audio sample."""
    frames = []
    for path in screenshot_paths[:SCREENSHOTS]:
        try:
            # Final screenshots are WebP; the AI analysis uses the JPG twin
            # (same frame) because the requests are sent as image/jpeg.
            jpg = Path(path).with_suffix(".jpg")
            frames.append((jpg if jpg.exists() else Path(path)).read_bytes())
        except OSError as ex:
            log(f"  Analysis screenshot skipped: {ex}")

    if not frames:
        raise RuntimeError("No screenshots available for AI analysis.")

    audio = outdir / "an_audio.mp3"
    start = dur * 0.10
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", str(AUDIO_MINUTES * 60),
         "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(audio)])
    return frames, audio.read_bytes()


def quick_analysis_inputs(src, dur, outdir, count=None):
    """Light-weight inputs for info extraction BEFORE the heavy video work:
    evenly spaced JPG frames + an audio sample (no scene detection, no WebP)."""
    count = count or max(8, SCREENSHOTS * 2)
    frames = []
    qdir = outdir / "quick_frames"
    qdir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        t = dur * (0.06 + 0.88 * i / max(1, count - 1))
        jpg = qdir / f"q_{i:02d}.jpg"
        try:
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", src,
                 "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2", "-q:v", "3", str(jpg)])
            if jpg.exists() and jpg.stat().st_size > 0:
                frames.append(jpg.read_bytes())
        except Exception as ex:  # noqa
            log(f"  Quick frame skipped: {ex}")
    if not frames:
        raise RuntimeError("No frames available for AI analysis.")

    audio = outdir / "an_audio.mp3"
    start = dur * 0.10
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", str(AUDIO_MINUTES * 60),
         "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(audio)])
    return frames, audio.read_bytes()


LONG_SIDE = {144: 256, 180: 320, 240: 426, 360: 640, 480: 854, 720: 1280,
             1080: 1920, 1440: 2560, 2160: 3840}   # long-side pixels for each quality label


def _probe_av(src):
    """Return (video_codec, pix_fmt, audio_codec, audio_channels) of the source."""
    def one(sel, entries):
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", sel,
                 "-show_entries", "stream=" + entries, "-of", "json", src],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=120)
            return (json.loads(r.stdout).get("streams") or [{}])[0]
        except Exception:  # noqa
            return {}
    v = one("v:0", "codec_name,pix_fmt")
    a = one("a:0", "codec_name,channels")
    return (v.get("codec_name", ""), v.get("pix_fmt", ""),
            a.get("codec_name", ""), int(a.get("channels") or 2))


def transcode(src, target, w, h, out):
    """target = length of the SHORT side (480/720/1080): works for landscape and vertical.
    Quality rules: original audio is kept (copied if AAC, otherwise re-encoded at a
    high bitrate with the SAME channel layout); video is copied untouched when the
    size already matches the source, otherwise encoded with a low (high-quality) CRF."""
    crf = CRF.get(target, 21)
    same_size = False
    if w >= h:
        if target <= h * 1.05:
            vf = f"scale=-2:{target}"                       # normal case: height = target
            same_size = (target == h)
        else:
            lw = LONG_SIDE.get(target, round(target * 16 / 9 / 2) * 2)
            vf = f"scale={lw}:-2"                           # widescreen: width-based
            same_size = (lw == w)
    else:
        vf = f"scale={target}:-2"
        same_size = (target == w)

    vcodec, pix_fmt, acodec, ach = _probe_av(src)

    copy_video = (PLAY_COPY_ORIGINAL and same_size
                  and vcodec == "h264" and pix_fmt == "yuv420p")
    if copy_video:
        log(f"  {target}p: source is H.264 at this size -> video copied (original quality).")
        vopts = ["-c:v", "copy"]
    else:
        vopts = ["-vf", vf, "-pix_fmt", "yuv420p",
                 "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(crf)]

    if acodec == "aac":
        aopts = ["-c:a", "copy"]                            # untouched original audio
    else:
        kbps = 128 if ach <= 1 else (192 if ach == 2 else 64 * ach)   # 5.1 -> 384k
        aopts = ["-c:a", "aac", "-b:a", f"{kbps}k"]         # keeps channel layout

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats", "-i", src,
           "-map", "0:v:0", "-map", "0:a?"] + vopts + aopts
    # +faststart only exists for the mp4/mov muxers (not mkv).
    if out.lower().endswith((".mp4", ".mov", ".m4v")):
        cmd += ["-movflags", "+faststart"]
    run(cmd + [out])


# ---------- menu labels (read from the live blog) ----------
FALLBACK_LABELS = [
    "Bollywood Content", "Desi Junction", "Dual Audio", "Hindi Dubbed", "Hindi TV Shows",
    "Web Series", "WWE", "Hollywood Movies", "Malayalam Movies", "Marathi Movies",
    "Mobile Movies", "Multi Audio", "Pakistani Movies", "PC Games", "Pre Release",
    "Punjabi Movies", "Single Video Songs", "Tamil Movies", "Telugu Movies", "Trailers",
    "Uncategorized",
]
BLOCKED_LABEL = re.compile(r"18\+|adult|xxx|hevc|x265", re.I)


def parse_labels(page):
    found = re.findall(r"/search/label/([^\"'?&#<>\s/]+)", page)
    labels, seen = [], set()
    for f in found:
        name = urllib.parse.unquote_plus(f).strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            labels.append(name)
    return labels


LABELS_CACHE = STATE_DIR / "labels.json"
LABELS_CACHE_HOURS = 24


def _read_labels_cache():
    try:
        d = json.loads(LABELS_CACHE.read_text(encoding="utf-8"))
        return d.get("labels", []), float(d.get("time", 0))
    except Exception:  # noqa
        return [], 0.0


def _write_labels_cache(labels):
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_CACHE.write_text(
            json.dumps({"time": time.time(), "labels": labels}, ensure_ascii=False),
            encoding="utf-8")
    except Exception as e:  # noqa
        log(f"  Could not save labels cache: {e}")


def get_site_labels(max_tries=3):
    """Read the real label names from the blog's menu.

    Uses a cached copy (state/labels.json) so the blog is only contacted about
    once a day. On errors such as HTTP 429 it retries a few times, then uses
    the old cache, and only then the built-in list."""
    cached, cached_at = _read_labels_cache()
    if len(cached) >= 3 and (time.time() - cached_at) < LABELS_CACHE_HOURS * 3600:
        log(f"  Using cached blog labels ({len(cached)})")
        labels = cached
    else:
        labels = []
        for attempt in range(1, max_tries + 1):
            try:
                url = blogger.blogs().get(blogId=BLOG_ID).execute()["url"]
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
                labels = parse_labels(page)
                log(f"  Found {len(labels)} labels on the blog")
                if len(labels) >= 3:
                    _write_labels_cache(labels)
                break
            except Exception as e:  # noqa
                wait = min(20 * attempt, 60)
                retry_after = getattr(getattr(e, "headers", None), "get", lambda *_: None)("Retry-After")
                if retry_after and str(retry_after).isdigit():
                    wait = min(max(wait, int(retry_after)), 120)
                if attempt == max_tries:
                    log(f"  Could not read blog labels after {max_tries} tries: {e}.")
                else:
                    log(f"  Could not read blog labels (try {attempt}/{max_tries}): {e} -> retrying in {wait}s")
                    time.sleep(wait)
        if len(labels) < 3 and len(cached) >= 3:
            log(f"  Using old cached labels ({len(cached)})")
            labels = cached
    if len(labels) < 3:
        log("  Using built-in label list")
        have = {l.lower() for l in labels}
        labels += [l for l in FALLBACK_LABELS if l.lower() not in have]
    labels = [l for l in labels if not BLOCKED_LABEL.search(l)]
    return labels[:60]


def pick_labels(raw, site_labels):
    canon = {l.lower(): l for l in site_labels}
    out = []
    for r in raw or []:
        c = canon.get(str(r).strip().lower())
        if c and c not in out:
            out.append(c)
    return out[:4]


# ---------- Gemini ----------
def as_paragraphs(v):
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [p.strip() for p in re.split(r"\n\s*\n", str(v or "")) if p.strip()]


YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")


def find_year(filename):
    """Release year only if the file name contains one (e.g. 'My Film 2019.mp4'), else None."""
    m = YEAR_RE.search(Path(filename).stem)
    return int(m.group(1)) if m else None


def clean_hint(filename):
    """Extract a useful movie-title hint from a messy video filename.

    Keep real title words, but remove the common release/encoding/social
    metadata that frequently gets appended to uploaded movie files.
    """
    t = Path(filename).stem
    t = re.sub(r"[#@]\S+", " ", t)
    t = re.sub(r"[_.]+", " ", t)
    t = re.sub(r"[\[\]{}()]+", " ", t)

    # Remove common technical/release metadata without touching normal title words.
    metadata = r"""\b(?:480p|576p|720p|1080p|1440p|2160p|4k|8k|x264|x265|h264|h265|hevc|av1|aac|ac3|ddp|dd|5\.1|2\.0|10bit|8bit|hdr|sdr|mp4|mkv|avi|mov|webm|m4v|bluray|blu[- ]?ray|web[- ]?dl|web[- ]?rip|webrip|brrip|hdrip|dvdrip|camrip|proper|repack|remux|yts|rarbg|hindi|english|tamil|telugu|malayalam|kannada|bengali|dual[ -]?audio|multi[ -]?audio|dubbed|subbed|subs|eng[ -]?sub|movie|full[ -]?movie|watch[ -]?online|download)\b"""
    t = re.sub(metadata, " ", t, flags=re.I)
    t = YEAR_RE.sub(" ", t)
    t = re.sub(r"[^\w\s'&:-]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip(" -_:|")
    return t


def filename_title_candidate(filename):
    """Return a strong title candidate only when the filename contains one."""
    raw = Path(filename).stem
    cleaned = clean_hint(filename)
    if not cleaned:
        return ""

    # Social/reel filenames are not reliable movie-title evidence.
    low = cleaned.lower()
    bad = {"wait for it", "instagram reels", "reels", "explore page",
           "viral reels", "content creator", "trending reels", "parrot skit",
           "video", "vid", "file", "movie", "film", "untitled", "new video",
           "source", "original", "output", "final", "clip"}
    if low in bad or len(cleaned.split()) > 10:
        return ""
    # Generic camera/app names such as "VID 20240101", "video 3", "IMG 0012".
    if re.fullmatch(r"(?:vid|video|img|clip|file|movie|film)?[\s_-]*\d[\d\s_-]*", low):
        return ""

    # A usable title is normally 1-8 words after metadata removal.
    words = cleaned.split()
    if 1 <= len(words) <= 8:
        return cleaned
    return ""


def _gemini_multimodal_json_rest(model, prompt, frames, audio_bytes=None):
    """Gemini multimodal JSON call through REST; avoids SDK AFC warnings."""
    parts = [{"text": prompt}]
    for b in frames:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(b).decode("ascii"),
            }
        })
    if audio_bytes:
        parts.append({
            "inline_data": {
                "mime_type": "audio/mp3",
                "data": base64.b64encode(audio_bytes).decode("ascii"),
            }
        })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(model, safe="")
        + ":generateContent?key="
        + urllib.parse.quote(GEMINI_KEY, safe="")
    )
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "MovieBot/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        response = json.loads(resp.read().decode("utf-8", "replace"))

    texts = []
    for candidate in response.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("text"):
                texts.append(part["text"])
    text = "\n".join(texts).strip()
    if not text:
        raise RuntimeError("Gemini returned no text.")
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip())


@disk_cache("gemini_imdb", 14)
def _gemini_imdb_lookup(title, year=None, language=""):
    """Look up an IMDb rating through Gemini Google Search grounding.

    We only accept a rating when Gemini returns an IMDb source URL and the
    matched title/year are reasonably consistent. Otherwise return N/A.
    """
    title = str(title or "").strip()
    if not title:
        return "N/A"

    year_text = str(year) if year else ""
    prompt = f"""Search the web for the exact movie/title below and verify its IMDb rating.
Movie title: {title}
Release year: {year_text or 'unknown'}
Language: {language or 'unknown'}

IMPORTANT:
- Use ONLY the official IMDb website (imdb.com) as the source for the rating.
- Do NOT guess or invent a rating.
- Match the title and release year carefully. If the title/year cannot be confidently matched,
  return rating as N/A.
- If IMDb has no displayed rating, return N/A.
- Return ONLY JSON with exactly these keys:
  matched_title: exact IMDb title if found, otherwise "",
  matched_year: year if found, otherwise null,
  rating: IMDb aggregate rating such as "7.2/10", otherwise "N/A",
  imdb_url: official IMDb title URL if found, otherwise ""
"""

    models = []
    for m in [
        os.environ.get("IMDB_GEMINI_MODEL", "gemini-2.5-flash"),
        GEMINI_MODEL,
        "gemini-2.5-flash",
    ]:
        if m and m not in models:
            models.append(m)

    for model in models:
        for attempt in range(1, 3):
            try:
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "tools": [{"google_search": {}}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.0,
                    },
                }
                endpoint = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    + urllib.parse.quote(model, safe="")
                    + ":generateContent?key="
                    + urllib.parse.quote(GEMINI_KEY, safe="")
                )
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "MovieBot/1.0"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    response = json.loads(resp.read().decode("utf-8", "replace"))

                texts = []
                imdb_sources = []
                for candidate in response.get("candidates") or []:
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        if part.get("text"):
                            texts.append(part["text"])
                    gm = candidate.get("groundingMetadata") or candidate.get("grounding_metadata") or {}
                    for chunk in gm.get("groundingChunks") or gm.get("grounding_chunks") or []:
                        web = chunk.get("web") or {}
                        uri = web.get("uri") or web.get("url") or ""
                        if "imdb.com" in uri.lower():
                            imdb_sources.append(uri)

                text = "\n".join(texts).strip()
                if not text:
                    raise RuntimeError("IMDb lookup returned no text")
                data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip())

                matched_title = str(data.get("matched_title") or "").strip()
                matched_year = data.get("matched_year")
                rating = str(data.get("rating") or "N/A").strip()
                imdb_url = str(data.get("imdb_url") or "").strip()
                if not imdb_url and imdb_sources:
                    imdb_url = imdb_sources[0]

                # Strict validation: the URL must actually come from Gemini's
                # IMDb grounding result, and the rating must be on IMDb's 1-10 scale.
                normalized_returned = imdb_url.split("?")[0].rstrip("/").lower()
                grounded_match = any(
                    normalized_returned == src.split("?")[0].rstrip("/").lower()
                    for src in imdb_sources
                ) if imdb_sources else False
                m = re.fullmatch(r"(?:10(?:\.0)?|[1-9](?:\.[0-9])?)/10", rating)
                if not grounded_match or "imdb.com" not in imdb_url.lower() or not m:
                    log(f"  IMDb lookup: no verified rating for '{title}'")
                    return "N/A"

                # Guard against an unrelated same-name result.
                ratio = difflib.SequenceMatcher(
                    None, re.sub(r"\W+", "", title.lower()),
                    re.sub(r"\W+", "", matched_title.lower())
                ).ratio() if matched_title else 0.0
                year_ok = True
                if year and matched_year:
                    try:
                        year_ok = int(matched_year) == int(year)
                    except Exception:
                        year_ok = False

                if ratio < 0.72 or not year_ok:
                    log(f"  IMDb lookup: title/year mismatch for '{title}' -> '{matched_title}' ({matched_year})")
                    return "N/A"

                log(f"  IMDb verified: {matched_title} ({matched_year or year_text}) = {rating}")
                return rating

            except urllib.error.HTTPError as ex:
                if ex.code in (408, 429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                log(f"  IMDb lookup via Gemini failed on {model}: HTTP {ex.code}")
                break
            except Exception as ex:
                log(f"  IMDb lookup via Gemini failed on {model}: {ex}")
                break

    return "N/A"


def normalize_movie_title(value, fallback=""):
    """Keep the Blogger title short, clean and poster-like.

    The AI may occasionally add marketing words even when asked not to.
    Strip only obvious metadata/marketing suffixes; do not rewrite genuine
    movie names.
    """
    text = html.unescape(str(value or "")).strip()
    text = text.strip('\"\'`“”‘’')
    text = YEAR_RE.sub(" ", text)
    text = re.sub(
        r"\s*(?:[-|:–—]\s*)?(?:full\s+movie|movie\s+full|watch\s+online|online\s+watch|download|official\s+trailer|trailer)\s*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" -|:–—")

    if not text or text.lower() in {"untitled", "untitled film", "movie", "film"}:
        text = clean_hint(fallback) if fallback else "Untitled Film"

    return text[:120].strip()


@disk_cache("gemini_facts", 90)
def _gemini_title_facts_lookup(title, current_year=None, current_language=""):
    """Fallback web verification for missing year/audio-language metadata."""
    title = str(title or "").strip()
    if not title:
        return {}
    prompt = f"""Find factual metadata for this exact film: {title}

Use Google Search and return ONLY JSON:
{{
  "release_year": 4-digit original release year or null,
  "languages": ["language1", "language2"],
  "original_language": "primary original language or Unknown"
}}
Rules: match the exact film; include only actual spoken/audio languages, not subtitle languages; do not guess.
"""
    for model in [GEMINI_MODEL, "gemini-flash-latest"]:
        if not model:
            continue
        try:
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "tools": [{"google_search": {}}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0},
            }
            endpoint = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + urllib.parse.quote(model, safe="")
                + ":generateContent?key=" + urllib.parse.quote(GEMINI_KEY, safe="")
            )
            req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "MovieBot/1.0"}, method="POST")
            with urllib.request.urlopen(req, timeout=90) as resp:
                response = json.loads(resp.read().decode("utf-8", "replace"))
            texts = []
            for candidate in response.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if part.get("text"):
                        texts.append(part["text"])
            if not texts:
                continue
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", "\n".join(texts).strip(), flags=re.I).strip()
            data = json.loads(raw)
            out = {}
            if not current_year:
                try:
                    y = int(data.get("release_year"))
                    if 1888 <= y <= 2100:
                        out["release_year"] = y
                except (TypeError, ValueError):
                    pass
            if not current_language:
                langs = _title_languages(" | ".join(str(x) for x in (data.get("languages") or [])))
                if langs:
                    out["language"] = " - ".join(langs)
            original = str(data.get("original_language") or "").strip()
            if original and original.lower() not in {"unknown", "n/a", "none", "null"}:
                out["original_language"] = original
            if out:
                log("  Title facts verified by Gemini search:", out)
                return out
        except Exception as ex:
            log(f"  Title-facts lookup failed on {model}: {ex}")
    return {}


def _gemini_audio_language_fallback(audio_bytes):
    """Identify spoken language from the extracted audio without relying on frames."""
    if not audio_bytes:
        return {}
    prompt = """Listen to the supplied movie audio carefully and identify the language(s) actually spoken.
Return ONLY JSON in exactly this form:
{
  "languages": ["Hindi", "English"],
  "primary_language": "Hindi"
}
Rules:
- Identify spoken/dialogue languages from the audio itself.
- Do NOT count subtitles, captions, background music, or metadata.
- If the audio is dubbed, report the language actually spoken in this audio.
- Use common English language names only (Hindi, English, Kannada, Telugu, Tamil, Malayalam, Bengali, Marathi, Punjabi, Urdu, etc.).
- If uncertain, return an empty languages list and primary_language "Unknown".
"""
    models = []
    for model in _gem_models():
        if model and model not in models:
            models.append(model)
    for model in models:
        try:
            payload = {
                "contents": [{"role": "user", "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "audio/mp3",
                        "data": base64.b64encode(audio_bytes).decode("ascii"),
                    }},
                ]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.0,
                },
            }
            endpoint = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + urllib.parse.quote(model, safe="")
                + ":generateContent?key=" + urllib.parse.quote(GEMINI_KEY, safe="")
            )
            req = urllib.request.Request(
                endpoint, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "MovieBot/1.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                response = json.loads(resp.read().decode("utf-8", "replace"))
            texts = []
            for candidate in response.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if part.get("text"):
                        texts.append(part["text"])
            if not texts:
                continue
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", "\n".join(texts).strip(), flags=re.I).strip()
            data = json.loads(raw)
            langs = _title_languages(" | ".join(str(x) for x in (data.get("languages") or [])))
            primary = str(data.get("primary_language") or "").strip()
            if primary.lower() in {"unknown", "n/a", "none", "null"}:
                primary = ""
            if langs or primary:
                if primary and not langs:
                    langs = [primary]
                log("  Spoken-language check from audio:", langs, "primary=", primary or "Unknown")
                return {"language": " - ".join(langs), "original_language": primary}
        except Exception as ex:
            log(f"  Gemini audio-language fallback failed on {model}: {ex}")
    return {}


def _gemini_video_language_fallback(filename_hint, frames, audio_bytes):
    """Dedicated audio/visual language check when the main analysis says Unknown."""
    prompt = f"""Listen carefully to the supplied audio sample and inspect the movie frames.
File hint: {clean_hint(filename_hint)}
Return ONLY JSON: {{"languages":["Hindi","English"],"original_language":"Hindi"}}
Include only languages actually spoken in the supplied audio. Do not count subtitles or captions.
If uncertain, return an empty languages list and original_language "Unknown".
"""
    models = []
    for model in _gem_models():
        if model and model not in models:
            models.append(model)
    for model in models:
        try:
            data = _gemini_multimodal_json_rest(model, prompt, frames, audio_bytes)
            langs = _title_languages(" | ".join(str(x) for x in (data.get("languages") or [])))
            original = str(data.get("original_language") or "").strip()
            if original.lower() in {"unknown", "n/a", "none", "null"}:
                original = ""
            if langs or original:
                return {"language": " - ".join(langs), "original_language": original}
        except Exception as ex:
            log(f"  Gemini video-language fallback failed on {model}: {ex}")
    return {}


# ---------- TMDB + Gemini synopsis rewrite ----------
def _norm_title(t):
    t = str(t or "").lower().replace("&", " and ")
    t = re.sub(r"\bcease[\s\-]+fire\b", "ceasefire", t)
    t = re.sub(r"\bpart[\s\-]*(one|1)\b", "part 1", t)
    t = re.sub(r"\bpart[\s\-]*(two|2)\b", "part 2", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def title_score(a, b):
    """0..1 similarity that ignores punctuation, word order and 'Cease Fire' vs
    'Ceasefire' (e.g. 'Salaar Cease Fire Part 1' == 'Salaar: Part 1 - Ceasefire')."""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na.replace(" ", ""), nb.replace(" ", "")).ratio()
    ta, tb = set(na.split()), set(nb.split())
    tok = len(ta & tb) / max(len(ta), len(tb))
    return max(seq, tok)


def title_variants(title):
    """Search spellings to try, best first (full title, cleaned, without part/
    chapter, first words only)."""
    base = re.sub(r"\s+", " ", str(title or "")).strip()
    n = _norm_title(base)
    no_part = re.sub(r"\b(part|chapter|vol|volume)\s*\d+\b", " ", n)
    no_part = re.sub(r"\s+", " ", no_part).strip()
    words = n.split()
    out = [base, n, no_part, " ".join(words[:2]), words[0] if words else ""]
    seen, res = set(), []
    for v in out:
        if v and v.lower() not in seen:
            seen.add(v.lower())
            res.append(v)
    return res


def _tmdb_get(path, params=None):
    params = dict(params or {})
    headers = {"Accept": "application/json", "User-Agent": "MovieBot/1.0"}
    if TMDB_READ_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_READ_TOKEN}"
    else:
        params["api_key"] = TMDB_API_KEY
    url = "https://api.themoviedb.org/3" + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


@disk_cache("tmdb", 30)
def tmdb_lookup(title, year=None, imdb_id=None):
    """Return TMDB facts for the movie, or {} on any problem (never raises)."""
    if not (TMDB_API_KEY or TMDB_READ_TOKEN):
        return {}
    try:
        best = None
        if imdb_id:
            found = retry(lambda: _tmdb_get(
                f"/find/{imdb_id}", {"external_source": "imdb_id"}), tries=2)
            hits = found.get("movie_results") or []
            best = hits[0] if hits else None
        if not best and title:
            pool = {}
            for q in title_variants(title):
                for with_year in ((True, False) if year else (False,)):
                    params = {"query": q, "include_adult": "false"}
                    if with_year:
                        params["year"] = year
                    try:
                        res = retry(lambda: _tmdb_get("/search/movie", params), tries=2).get("results") or []
                    except Exception as ex:  # noqa
                        log(f"  TMDB search '{q}' failed: {ex}")
                        res = []
                    for r in res[:8]:
                        pool.setdefault(r.get("id"), r)
                if len(pool) >= 3 and any(
                        title_score(title, r.get("title")) >= 0.9 for r in pool.values()):
                    break

            def cand_score(r):
                sc = max(title_score(title, r.get("title")),
                         title_score(title, r.get("original_title")))
                try:
                    ry = int(str(r.get("release_date") or "")[:4])
                    if year and abs(ry - int(year)) <= 1:
                        sc += 0.05
                    elif year and abs(ry - int(year)) > 2:
                        sc -= 0.15
                except ValueError:
                    pass
                return sc

            ranked = sorted(pool.values(),
                            key=lambda r: (cand_score(r), r.get("popularity") or 0),
                            reverse=True)
            best = ranked[0] if ranked else None
            if not best or cand_score(best) < 0.72:
                log(f"  TMDB: no confident match for '{title}'"
                    + (f" (closest: '{best.get('title')}')" if best else ""))
                return {}
        if not best:
            return {}

        # Also ask TMDB for the movie's poster images (so we can pick the
        # best one). Languages: English + the movie's original language
        # + textless (null).
        orig = best.get("original_language") or ""
        d = retry(lambda: _tmdb_get(
            f"/movie/{best['id']}",
            {"append_to_response": "external_ids,images",
             "include_image_language": f"en,{orig},null"}), tries=2)
        y = None
        try:
            y = int(str(d.get("release_date") or "")[:4])
        except ValueError:
            pass

        # Best poster first: real 2:3 ratio, decent width, prefer English /
        # original-language text, then highest vote, then widest.
        posters = (d.get("images") or {}).get("posters") or []
        posters = [p for p in posters
                   if p.get("file_path")
                   and 0.62 <= (p.get("aspect_ratio") or 0) <= 0.72
                   and (p.get("width") or 0) >= 500]
        posters.sort(key=lambda p: ((p.get("iso_639_1") in ("en", orig)),
                                    p.get("vote_average") or 0,
                                    p.get("width") or 0), reverse=True)
        poster_urls = ["https://image.tmdb.org/t/p/original" + p["file_path"]
                       for p in posters[:4]]
        if not poster_urls and d.get("poster_path"):
            poster_urls = ["https://image.tmdb.org/t/p/original" + d["poster_path"]]

        out = {
            "id": d.get("id"),
            "title": d.get("title"),
            "overview": str(d.get("overview") or "").strip(),
            "year": y,
            "rating": d.get("vote_average"),
            "genres": [g["name"] for g in d.get("genres") or [] if g.get("name")],
            "imdb_id": (d.get("external_ids") or {}).get("imdb_id"),
            "runtime": d.get("runtime"),
            "original_language": d.get("original_language"),
            "poster_urls": poster_urls,
        }
        log(f"  TMDB matched: {out['title']} ({out['year']}) id={out['id']} "
            f"posters={len(poster_urls)}")
        return out
    except Exception as ex:
        log(f"  TMDB lookup failed (limit/network/key?): {ex}")
        return {}


def _omdb_get(params):
    params = dict(params)
    params["apikey"] = OMDB_API_KEY
    url = "https://www.omdbapi.com/?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "MovieBot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


@disk_cache("omdb", 14)
def omdb_lookup(title, year=None, imdb_id=None):
    """Return OMDb facts (incl. real IMDb rating), or {} on any problem (never raises)."""
    if not OMDB_API_KEY:
        return {}
    try:
        def clean(v):
            v = str(v or "").strip()
            return "" if v.upper() == "N/A" else v

        d = None
        if imdb_id:
            d = retry(lambda: _omdb_get({"i": imdb_id, "plot": "full"}), tries=2)
            if str(d.get("Response")).lower() != "true":
                log(f"  OMDb: {d.get('Error') or 'no result'}")
                d = None
        elif title:
            # 1) exact-title tries with several spellings
            for q in title_variants(title):
                for with_year in ((True, False) if year else (False,)):
                    params = {"t": q, "type": "movie", "plot": "full"}
                    if with_year:
                        params["y"] = year
                    r = retry(lambda: _omdb_get(params), tries=2)
                    if str(r.get("Response")).lower() == "true" and \
                            title_score(title, clean(r.get("Title"))) >= 0.72:
                        d = r
                        break
                if d:
                    break
            # 2) search list, pick the closest title, then fetch it by IMDb id
            if not d:
                best_id, best_sc = None, 0.0
                for q in title_variants(title)[:4]:
                    r = retry(lambda: _omdb_get({"s": q, "type": "movie"}), tries=2)
                    for it in (r.get("Search") or [])[:10]:
                        sc = title_score(title, it.get("Title"))
                        try:
                            if year and abs(int(str(it.get("Year"))[:4]) - int(year)) <= 1:
                                sc += 0.05
                        except ValueError:
                            pass
                        if sc > best_sc:
                            best_id, best_sc = it.get("imdbID"), sc
                if best_id and best_sc >= 0.72:
                    r = retry(lambda: _omdb_get({"i": best_id, "plot": "full"}), tries=2)
                    if str(r.get("Response")).lower() == "true":
                        d = r
        if not d:
            log(f"  OMDb: no confident match for '{title}'")
            return {}

        rating = clean(d.get("imdbRating"))
        if not re.fullmatch(r"(?:10(?:\.0)?|[1-9](?:\.[0-9])?)", rating):
            rating = ""
        y = None
        m = re.search(r"\d{4}", clean(d.get("Year")))
        if m:
            y = int(m.group(0))

        # OMDb returns a small SX300 poster URL; strip the size suffix to
        # get the full-size IMDb poster.
        poster = clean(d.get("Poster"))
        if poster:
            poster = re.sub(r"\._V1_.*?(\.\w+)$", r"._V1_\1", poster)

        out = {
            "title": clean(d.get("Title")),
            "year": y,
            "imdb_id": clean(d.get("imdbID")),
            "imdb_rating": f"{rating}/10" if rating else "",
            "plot": clean(d.get("Plot")),
            "genres": [g.strip() for g in clean(d.get("Genre")).split(",") if g.strip()],
            "poster_url": poster,
        }
        log(f"  OMDb matched: {out['title']} ({out['year']}) rating={out['imdb_rating'] or 'N/A'}")
        return out
    except Exception as ex:
        log(f"  OMDb lookup failed (limit/network/key?): {ex}")
        return {}


@disk_cache("gemini_synopsis", 90)
def gemini_rewrite_synopsis(title, year, overview):
    """Rewrite a TMDB overview with Gemini. Returns [] if every model fails."""
    prompt = f"""Rewrite this movie overview as an original synopsis for a film blog.
Movie: {title} {f'({year})' if year else ''}
Source overview: {overview}

Rules:
- Your own words, natural English, 2 compact paragraphs, about 120-180 words total.
- Use ONLY facts in the source overview. Do not invent characters, events or endings.
- Spoiler-light. No piracy words, no mention of TMDB, AI or this website.
Return ONLY JSON: {{"synopsis": ["paragraph 1", "paragraph 2"]}}"""
    models = []
    for m in _gem_models():
        if m and m not in models:
            models.append(m)
    for model in models:
        try:
            data = _gemini_multimodal_json_rest(model, prompt, [], None)
            paras = as_paragraphs(data.get("synopsis") or data.get("description"))
            if paras:
                log(f"  Synopsis rewritten by Gemini ({model})")
                return paras[:2]
        except Exception as ex:
            log(f"  Gemini synopsis rewrite failed on {model}: {ex}")
    return []


def analyze(filename_hint, frames, audio_bytes, site_labels):
    hint = clean_hint(filename_hint)
    year = find_year(filename_hint)
    prompt = f"""You are a film writer and metadata editor for an ORIGINAL movie blog.
You get 12 frames spread across the film and an audio sample.
File name hint (may be messy): "{hint}". Language hint (may be empty): "{LANGUAGE_HINT}".
Director name (may be empty): "{DIRECTOR_NAME}".
Manual title (if supplied): "{MANUAL_TITLE}".

Rules:
- Write everything in your own words, in natural English. Never copy text from any website, film or review.
- Base movie facts ONLY on what you can actually see/hear and the filename hint. If uncertain, use a safe general value.
- Do not invent cast, crew, awards, festivals, box office, IMDb pages, or exact plot facts.
- Identify the actual film title whenever it is visible in a frame/title card, opening/closing title, or clearly present in the filename.
- Treat a recognizable filename title as a strong candidate, but verify it against the supplied frames before changing it.
- If a genuine title is visible, preserve its wording; only normalize capitalization/spacing. Do NOT invent a different title just to make it sound more attractive.
- Never use a character name, actor name, tagline, scene description, genre, or generic phrase as the movie title when a real title can be identified.
- Title must be 1-8 words, with no hashtags, emojis, year, language, quality, file size, "trending reels", "watch online", "download", "full movie", or "official trailer" text.
- Write the description as a proper FULL-MOVIE synopsis, even when the uploaded source is only a short clip or a short excerpt. Do NOT describe only the selected scene (for example, do not start with "A powerful 30-second clip...").
- When the movie title is reliably identified from the filename or supplied visual/audio evidence, write the synopsis for the whole movie: introduce the protagonist(s), central premise, setting, major conflict and overall story progression. You may use well-known factual plot knowledge associated with the identified movie, but do not invent characters, events, relationships or endings.
- Keep the synopsis spoiler-light: explain the movie's main journey and conflict without revealing the final twist, ending or major resolution.
- Do not mention that AI analyzed the movie, do not mention the filename, the uploaded clip length, screenshots, or the website.
- The website description should be 2 compact paragraphs, about 120-180 words total, natural, informative and engaging.
- Language should list the languages actually evident from the audio/filename when possible, for example "Hindi - English".
- Original language should be the primary/original spoken language when reasonably identifiable; otherwise "Unknown".
- Genres should be 1-3 suitable genres based on the film.
- Content rating should be one of "General audience", "Teen and above", "Mature audience".
- No piracy words in title/description/metadata (leaked, HD print, free download full movie, WEB-DL, dual audio, 300mb).

Return ONLY JSON with these keys:
  title: clean film title,
  release_year: 4-digit release year when it can be identified from the supplied video/filename/context, otherwise null,
  tagline: one short sentence, max 18 words,
  description: 2 compact paragraphs, about 120-180 words total, a full-movie spoiler-light synopsis even if the source video is only a short clip,
  imdb_rating: always return "N/A" here; IMDb will be verified separately through official IMDb search,
  language: display language(s), e.g. "Hindi - English",
  original_language: original/main language, e.g. "English",
  genres: list of 1-3 genres,
  content_rating: one of the allowed values,
  tags: list of up to 6 short keywords,
  labels: pick 1-4 categories that best fit this film, ONLY from this exact list
          (copy spelling exactly): {json.dumps(site_labels)}.
          Judge by language spoken, film industry/country, and type. Ignore encoding/file-format labels."""
    data = {}
    analysis_models = []
    for model in _gem_models():
        if model and model not in analysis_models:
            analysis_models.append(model)

    for model in analysis_models:
        for attempt in range(1, 5):
            try:
                data = _gemini_multimodal_json_rest(model, prompt, frames, audio_bytes)
                log("  Gemini model used:", model)
                break
            except urllib.error.HTTPError as ex:
                detail = ""
                try:
                    detail = ex.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                if ex.code in (408, 429, 500, 502, 503, 504) and attempt < 4:
                    wait = 5 * 3 ** (attempt - 1)   # 5s, 15s, 45s
                    log(f"  Gemini analysis {model}: HTTP {ex.code}; retrying in {wait}s ({attempt}/4)")
                    time.sleep(wait)
                    continue
                log(f"  Gemini analysis {model}: HTTP {ex.code}: {detail}")
                break
            except (urllib.error.URLError, TimeoutError) as ex:
                if attempt < 3:
                    wait = 2 ** attempt
                    log(f"  Gemini analysis {model}: temporary network error; retrying in {wait}s ({attempt}/3)")
                    time.sleep(wait)
                    continue
                log(f"  Gemini analysis {model}: network error after 3 attempts: {ex}")
                break
            except Exception as e:  # noqa
                log(f"  Gemini model {model} failed: {e}")
                break
        if data:
            break

    if not data:
        # Gemini failed: use the local Qwen3-VL fallback directly.
        try:
            local_dir = WORK / "local_ai_frames"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_paths = []
            for i, frame_bytes in enumerate(frames[:LOCAL_AI_MAX_IMAGES]):
                fp = local_dir / f"frame_{i}.jpg"
                fp.write_bytes(frame_bytes)
                local_paths.append(fp)
            data = _local_qwen_analyze(filename_hint, local_paths, site_labels)
            log("  Local Qwen3-VL metadata fallback succeeded.")
        except Exception as ex:
            log(f"  Local Qwen3-VL metadata fallback failed: {ex}")

    if not data:
        log("  Using deterministic fallback text.")

    desc = as_paragraphs(data.get("description"))
    if not desc:
        desc = as_paragraphs(data.get("synopsis"))
    if not desc:
        fallback_title = filename_title_candidate(filename_hint) or clean_hint(filename_hint) or "This film"
        desc = [
            f"{fallback_title} follows a central character whose life is shaped by the people, circumstances and conflict established in the story. The film develops its premise through the characters' goals, challenges and decisions, building toward a larger confrontation while maintaining its overall dramatic and cinematic tone.",
            "This synopsis is kept spoiler-light and avoids claiming specific events that could not be reliably established. It focuses on the film's central premise and story direction rather than describing only the short portion of the movie contained in the uploaded video.",
        ]

    faq = [f for f in (data.get("faq") or []) if isinstance(f, dict) and f.get("q") and f.get("a")]
    ai_title = normalize_movie_title(data.get("title"), "")
    if data.get("release_year") and not year:
        try:
            year = int(data.get("release_year"))
        except (TypeError, ValueError):
            pass
    filename_candidate = filename_title_candidate(filename_hint)
    if MANUAL_TITLE:
        final_title = normalize_movie_title(MANUAL_TITLE, filename_hint)
        log("  Title source: MANUAL_TITLE ->", final_title)
    elif filename_candidate:
        final_title = normalize_movie_title(filename_candidate, filename_hint)
        # If Gemini/Qwen produced an exact-looking title matching the filename,
        # keep the filename spelling; this prevents hallucinated replacement titles.
        log("  Title source: filename candidate ->", final_title)
    else:
        final_title = normalize_movie_title(ai_title, filename_hint)
        log("  Title source: AI/frame analysis ->", final_title)
    # Language must come from the actual video/audio when possible.
    # If the main multimodal analysis says Unknown, first ask Gemini using
    # the audio alone, then use the visual+audio check, and only then use
    # title-based web verification as a last factual fallback.
    current_lang = str(data.get("language") or "").strip()
    if not current_lang or current_lang.lower() in {"unknown", "n/a", "none", "null"}:
        try:
            lang_fallback = _gemini_audio_language_fallback(audio_bytes)
            if lang_fallback.get("language"):
                data["language"] = lang_fallback["language"]
            if lang_fallback.get("original_language"):
                data["original_language"] = lang_fallback["original_language"]
        except Exception as ex:
            log("  Audio-language fallback error:", ex)

    current_lang = str(data.get("language") or "").strip()
    if not current_lang or current_lang.lower() in {"unknown", "n/a", "none", "null"}:
        try:
            lang_fallback = _gemini_video_language_fallback(filename_hint, frames, audio_bytes)
            if lang_fallback.get("language"):
                data["language"] = lang_fallback["language"]
            if lang_fallback.get("original_language"):
                data["original_language"] = lang_fallback["original_language"]
        except Exception as ex:
            log("  Video-language fallback error:", ex)

    if MANUAL_TITLE and (not year or not str(data.get("language") or "").strip() or str(data.get("language")).strip().lower() in {"unknown", "n/a", "none", "null"}):
        try:
            facts = _gemini_title_facts_lookup(final_title, year, data.get("language") or "")
            if not year and facts.get("release_year"):
                year = facts["release_year"]
            if (not str(data.get("language") or "").strip() or str(data.get("language")).strip().lower() in {"unknown", "n/a", "none", "null"}) and facts.get("language"):
                data["language"] = facts["language"]
            if facts.get("original_language") and (not data.get("original_language") or str(data.get("original_language")).lower() == "unknown"):
                data["original_language"] = facts["original_language"]
        except Exception as ex:
            log("  Title-facts fallback error:", ex)

    # Final lightweight language fallback from the filename only when it
    # explicitly contains a language marker. This never guesses from the
    # movie title itself.
    final_lang = str(data.get("language") or LANGUAGE_HINT or "").strip()
    if not final_lang or final_lang.lower() in {"unknown", "n/a", "none", "null"}:
        lower_name = str(filename_hint or "").lower()
        found = []
        language_markers = [
            ("hindi", "Hindi"), ("english", "English"), ("kannada", "Kannada"),
            ("telugu", "Telugu"), ("tamil", "Tamil"), ("malayalam", "Malayalam"),
            ("bengali", "Bengali"), ("marathi", "Marathi"), ("punjabi", "Punjabi"),
            ("urdu", "Urdu"),
        ]
        for marker, label in language_markers:
            if re.search(r"(?<![a-z])" + re.escape(marker) + r"(?![a-z])", lower_name) and label not in found:
                found.append(label)
        if found:
            final_lang = " - ".join(found[:4])

    final_original = str(data.get("original_language") or "").strip()
    if not final_original or final_original.lower() in {"unknown", "n/a", "none", "null"}:
        # Keep a known spoken language rather than showing Unknown when that
        # is all we can establish reliably.
        final_original = _title_languages(final_lang)[0] if _title_languages(final_lang) else "Unknown"

    # ----- TMDB + OMDb: facts, real IMDb rating, posters, Gemini synopsis rewrite -----
    # Each source is optional and can fail (no key / daily limit / no match)
    # without stopping the other one:
    #   Synopsis: longest of TMDB overview / OMDb plot -> Gemini rewrite
    #             -> Gemini failed: raw overview -> no source: Gemini/Qwen text above
    #   Rating:   OMDb IMDb rating -> Gemini IMDb search -> N/A
    #   Posters:  TMDB best posters -> OMDb (IMDb) poster
    # Title is never changed by these sources.
    start_id = IMDB_ID if re.fullmatch(r"tt\d{6,10}", IMDB_ID) else ""
    tmdb = tmdb_lookup(final_title, year, start_id or None)
    omdb = omdb_lookup(final_title, year, start_id or None)
    imdb_id = start_id or tmdb.get("imdb_id") or omdb.get("imdb_id") or ""
    if imdb_id and imdb_id != start_id:
        # One source found the IMDb id: let the other one use it for an exact match.
        if not tmdb:
            tmdb = tmdb_lookup(final_title, year, imdb_id)
        if not omdb:
            omdb = omdb_lookup(final_title, year, imdb_id)

    if tmdb or omdb:
        year = year or tmdb.get("year") or omdb.get("year")
        genres = tmdb.get("genres") or omdb.get("genres")
        if genres:
            data["genres"] = genres[:3]
        sources = [t for t in (tmdb.get("overview"), omdb.get("plot")) if t]
        if sources:
            source_text = max(sources, key=len)
            rewritten = gemini_rewrite_synopsis(final_title, year, source_text)
            if rewritten:
                desc = rewritten
            else:
                log("  Gemini failed -> using original TMDB/OMDb text as synopsis")
                desc = as_paragraphs(source_text)[:2]

    # Official poster candidates for the thumbnail (best first).
    poster_urls = list(tmdb.get("poster_urls") or [])
    if omdb.get("poster_url"):
        poster_urls.append(omdb["poster_url"])
    poster_urls = list(dict.fromkeys(poster_urls))
    log(f"  Official poster candidates: {len(poster_urls)}")

    imdb_rating = omdb.get("imdb_rating") or "N/A"
    if imdb_rating == "N/A":
        log("  No OMDb IMDb rating; trying Gemini IMDb search...")
        imdb_rating = _gemini_imdb_lookup(final_title, year, final_lang)
    return {
        "title": final_title or "Untitled Film",
        "source_filename": filename_hint,
        "tagline": str(data.get("tagline") or "").strip(),
        "description": desc[:2],
        "imdb_rating": imdb_rating,
        "imdb_id": imdb_id,
        "language": final_lang or "Unknown",
        "original_language": final_original,
        "genres": [str(g).strip() for g in (data.get("genres") or ["Drama"]) if str(g).strip()][:3],
        "content_rating": str(data.get("content_rating") or "General audience").strip(),
        "tags": [str(t).strip() for t in (data.get("tags") or []) if str(t).strip()][:6],
        "release_year": year,
        "labels": pick_labels(data.get("labels"), site_labels),
        "poster_urls": poster_urls,
        # Kept for compatibility with any other code that may read these fields.
        "synopsis": desc[:2],
        "review": [],
        "themes": [],
        "faq": faq[:4],
    }


# ---------- post HTML ----------
def fmt_clock(sec):
    """Same format as the player JS: m:ss or h:mm:ss."""
    s_ = max(0, int(sec or 0))
    h_, m_, x_ = s_ // 3600, s_ % 3600 // 60, s_ % 60
    return (f"{h_}:{m_:02d}" if h_ else f"{m_}") + f":{x_:02d}"


def human(n):
    n = float(n)
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f}GB"
    return f"{n / 1024 ** 2:.0f}MB"


def fmt_runtime(sec):
    sec = int(sec)
    if sec < 60:
        return f"{sec} sec"
    m = sec // 60
    if m < 60:
        return f"{m} min"
    return f"{m // 60} h {m % 60} min"


def img_url(ref):
    """Images are hosted on Internet Archive (direct URL). Old Drive ids still work."""
    ref = str(ref)
    if ref.startswith("http"):
        return ref
    return f"https://lh3.googleusercontent.com/d/{ref}"


# The Download button now carries the Internet Archive URL in data-url.
# After the countdown the browser is sent straight to that URL.
TIMER_SCRIPT = """<script>
(function () {
  var WAIT = %d;
  var btns = document.querySelectorAll('a.mv-dl[data-url]');
  for (var i = 0; i < btns.length; i++) {
    (function (b) {
      var label = b.innerHTML;
      var busy = false;
      b.addEventListener('click', function (e) {
        e.preventDefault();
        if (busy) { return; }
        busy = true;
        var left = WAIT;
        b.style.opacity = '0.85';
        b.innerHTML = 'Please wait ' + left + ' seconds...';
        var t = setInterval(function () {
          left--;
          if (left > 0) {
            b.innerHTML = 'Please wait ' + left + ' seconds...';
            return;
          }
          clearInterval(t);
          b.innerHTML = 'Download starting...';
          window.location.href = b.getAttribute('data-url');
          setTimeout(function () {
            b.innerHTML = label;
            b.style.opacity = '1';
            busy = false;
          }, 6000);
        }, 1000);
      });
    })(btns[i]);
  }
})();
</script>""" % WAIT_SECONDS


def _post_colors(title):
    """Pick a stable-but-different color theme per movie post."""
    palettes = [
        {"heading": "#ffbf00", "rating": "#18e000", "label": "#e8e8e8", "value": "#f5f5f5", "lang": "#ff3030", "quality": "#ff3030"},
        {"heading": "#00d9ff", "rating": "#7dff2a", "label": "#ededed", "value": "#ffffff", "lang": "#ff4f81", "quality": "#ff4f81"},
        {"heading": "#ff6b35", "rating": "#65ff4d", "label": "#eeeeee", "value": "#ffffff", "lang": "#ff2f92", "quality": "#ff2f92"},
        {"heading": "#b56cff", "rating": "#48ff9b", "label": "#ededed", "value": "#ffffff", "lang": "#ff4d4d", "quality": "#ff4d4d"},
        {"heading": "#ffd166", "rating": "#39ff14", "label": "#f0f0f0", "value": "#ffffff", "lang": "#00d9ff", "quality": "#00d9ff"},
        {"heading": "#00e5a8", "rating": "#a8ff00", "label": "#ededed", "value": "#ffffff", "lang": "#ff3b30", "quality": "#ff3b30"},
        {"heading": "#ff8c42", "rating": "#00ff7f", "label": "#eeeeee", "value": "#ffffff", "lang": "#ff4d6d", "quality": "#ff4d6d"},
        {"heading": "#4dabf7", "rating": "#7cff00", "label": "#eeeeee", "value": "#ffffff", "lang": "#ff5c8a", "quality": "#ff5c8a"},
    ]
    digest = hashlib.sha256(str(title).encode("utf-8")).hexdigest()
    return palettes[int(digest[:8], 16) % len(palettes)]


def _title_languages(value):
    """Turn AI language text into clean title segments such as Hindi | English."""
    text = str(value or "").strip()
    if not text or text.lower() in {"unknown", "n/a", "none", "null"}:
        return []
    # Normalize common separators used by Gemini/Qwen.
    text = re.sub(r"\s*(?:-|–|—|/|,|&|\+)\s*", "|", text)
    parts = []
    for part in text.split("|"):
        part = re.sub(r"\s+", " ", part).strip(" .")
        if not part:
            continue
        if part.lower() in {"unknown", "n/a", "none", "null"}:
            continue
        if part.lower() not in {x.lower() for x in parts}:
            parts.append(part)
    return parts[:4]


def make_final_post_title(meta, outputs):
    """Build the complete Blogger title from the owner title + video metadata."""
    base = str(MANUAL_TITLE or meta.get("title") or "Untitled Film").strip()
    base = normalize_movie_title(base, meta.get("source_filename", ""))

    year = meta.get("release_year")
    year_text = ""
    if year:
        try:
            year_int = int(year)
            if 1888 <= year_int <= 2100:
                year_text = f" ({year_int})"
        except (TypeError, ValueError):
            pass

    languages = _title_languages(meta.get("language") or meta.get("original_language") or LANGUAGE_HINT)
    language_text = " | ".join(languages)

    quality_text = " | ".join(f"{int(h)}p" for h, *_ in outputs)

    lead = f"{base}{year_text}"
    if language_text:
        lead += f" – {language_text}"
    parts = [lead]
    if quality_text:
        parts.append(quality_text)
    parts.append("Watch Online & Download")
    return " | ".join(parts)


def build_html(meta, thumb_id, shot_ids, outputs, fps, dur, player_poster="",
               play_outputs=None):
    # outputs = list of (height, download_url, download_size_in_bytes, play_mp4_url)
    e = html.escape
    raw_title = str(MANUAL_TITLE or meta.get("title") or "").strip()
    if not raw_title:
        raw_title = str(filename_title_candidate(meta.get("source_filename", "")) or "Untitled Film").strip()
    title = e(raw_title)
    year = meta["release_year"]
    ytxt = f" ({year})" if year else ""
    lang_known = meta.get("language") and str(meta["language"]).lower() != "unknown"
    lang = e(str(meta.get("language") or "Unknown"))
    original_lang = e(str(meta.get("original_language") or "Unknown"))
    genres = ", ".join(e(g) for g in meta.get("genres", [])) or "Drama"
    fps_txt = fmt_fps(fps)
    qualities = " - ".join(f"{h}p" for h, *_ in outputs)
    sizes = " - ".join(human(o[2]) for o in outputs)
    colors = _post_colors(raw_title)

    # Premium compact Movie Info card.
    # Keep the reference-style fields, but remove the excessive vertical gaps.
    rating = str(meta.get("imdb_rating") or "N/A")

    def chip(text, accent):
        return (
            f'<span style="display:inline-block;padding:3px 8px;margin:2px 4px 2px 0;'
            f'border-radius:999px;border:1px solid {accent}66;background:{accent}18;'
            f'color:{accent};font-weight:800;font-size:12px;line-height:1.2;">{e(text)}</span>'
        )

    quality_chips = "".join(chip(f"{h}p", colors["quality"]) for h, *_ in outputs) or chip("N/A", colors["quality"])
    size_chips = "".join(chip(human(o[2]), colors["value"]) for o in outputs) or chip("N/A", colors["value"])

    def info_item(label, value, accent=None, full=False):
        value_color = accent or colors["value"]
        width = "100%" if full else "50%"
        return (
            f'<div style="box-sizing:border-box;width:{width};padding:5px 8px;min-width:0;">'
            f'<div style="font-size:11px;line-height:1.15;text-transform:uppercase;letter-spacing:.45px;'
            f'color:{colors["label"]};opacity:.72;margin-bottom:3px;">{e(label)}</div>'
            f'<div style="font-size:14px;line-height:1.35;color:{value_color};font-weight:700;overflow-wrap:anywhere;">{value}</div>'
            f'</div>'
        )

    # Movie Name is synced from the actual Blogger post title in the browser.
    # This is important because the owner may manually edit the Blogger post
    # title after the bot creates the draft. The initial value is only a
    # fallback for users who do not edit the title manually.
    movie_name_value = (
        f'<span class="mv-movie-name-value" data-mv-fallback="{e(raw_title, quote=True)}">'
        f'{title}</span>'
    )

    info_cells = [
        info_item("IMDb Rating", f'<span style="color:{colors["rating"]};">★ {e(rating)}</span>'),
        info_item("Movie Name", movie_name_value),
    ]
    if year:
        info_cells.append(info_item("Release Year", str(year)))
    if DIRECTOR_NAME:
        info_cells.append(info_item("Directed by", e(DIRECTOR_NAME)))
    if lang_known:
        info_cells.append(info_item("Language", lang, colors["lang"]))
    info_cells.append(info_item("Original Language", original_lang))
    info_cells.append(info_item("Runtime", fmt_runtime(dur)))
    info_cells.append(info_item("Genres", genres))
    info_cells.append(info_item("Content Advisory", e(meta["content_rating"])))
    info_cells.append(info_item("Frame Rate", f"{e(fps_txt)} fps"))
    info_cells.append(info_item("Quality", quality_chips, colors["quality"], full=True))
    info_cells.append(info_item("Size", size_chips, colors["value"], full=True))

    info_title = (
        f'<div style="text-align:center;color:{colors["heading"]};font-size:20px;line-height:1.2;'
        f'margin:18px 0 10px;font-weight:900;letter-spacing:.2px;">Movie Info</div>'
    )
    info_card = (
        f'<div style="width:100%;box-sizing:border-box;margin:0 auto 22px;padding:7px 4px;'
        f'border:1px solid rgba(255,255,255,.13);border-radius:12px;'
        f'background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.025));'
        f'box-shadow:0 7px 22px rgba(0,0,0,.24);">'
        f'<div style="display:flex;flex-wrap:wrap;align-items:stretch;">'
        + "".join(info_cells) +
        f'</div></div>'
    )

    btn = (
        "display:block;width:200px;max-width:82%;margin:0 auto 18px;padding:11px 8px;"
        "text-align:center;color:#fff;font-weight:800;font-size:14px;"
        "text-decoration:none;cursor:pointer;border-radius:6px;"
        "background:linear-gradient(90deg,#57a51c,#1f4fb4);"
        "box-shadow:0 8px 14px rgba(0,0,0,.45);"
    )
    head = (
        "text-align:center;color:#fff;font-size:15px;line-height:1.3;"
        "margin:16px 0 10px;font-weight:800"
    )
    hr = '<hr style="border:0;border-top:1px solid rgba(255,255,255,.35);margin:24px 0"/>'

    parts = []

    # 1) 2:3 poster thumbnail (WebP): TMDB/IMDb official poster first
    parts.append(
        f'<div style="text-align:center;margin:0 auto 18px;">'
        f'<img src="{img_url(thumb_id)}" alt="{e(seo_alt(meta, "movie poster"))}" '
        f'width="360" style="display:block;width:360px;max-width:72%;height:auto;'
        f'margin:0 auto;border-radius:8px;box-shadow:0 8px 22px rgba(0,0,0,.35);"/>'
        f'</div>'
    )

    # 2) Title
    parts.append(
        f'<h2 style="text-align:center;color:{colors["heading"]};font-size:24px;'
        f'line-height:1.3;margin:10px 0 28px;font-weight:800;">{title}</h2>'
    )

    # 3) Movie Info — same field structure/style as bot-9.py, with the
    # current verified IMDb rating added at the top.
    parts.append(info_title)
    parts.append(info_card)

    # 4) Custom player (plays the Internet Archive files) — right after Movie Info.
    parts.append(
        f'<h3 style="text-align:center;color:{colors["heading"]};font-size:22px;'
        f'margin:28px 0 18px;">Watch {title} Online</h3>'
    )
    # Player thumbnail = first frame of the video (falls back to screenshot 1
    # only for old saved states that have no first-frame poster yet).
    poster = img_url(player_poster) if player_poster else (
        img_url(shot_ids[0]) if shot_ids else "")
    player_label = meta.get("language") if lang_known else ""
    parts.append(
        '<div style="margin:0 auto 28px">'
        + build_player(play_outputs or outputs, player_label, poster, dur)
        + '</div>'
    )

    # 5) Screenshots — no description/review/info between player and screenshots.
    parts.append(
        f'<h3 style="text-align:center;color:{colors["heading"]};font-size:22px;'
        f'margin:28px 0 18px;">Screenshots</h3>'
    )
    for n, fid in enumerate(shot_ids, 1):
        parts.append(
            f'<div style="width:100%;max-width:1920px;margin:0 auto 18px;line-height:0;'
            f'padding:0;background:none;">'
            f'<img src="{img_url(fid)}" alt="{e(seo_alt(meta, f"movie screenshot {n}"))}" '
            'style="display:block;width:100%;height:auto;max-width:1920px;margin:0;padding:0;'
            'border:0;outline:0;box-shadow:none"/>'
            f'</div>'
        )

    # 6) Download buttons — directly after screenshots. Nothing else in between.
    # Each button links to the file on Internet Archive.
    parts.append(hr)
    parts.append(
        f'<h3 style="text-align:center;color:{colors["heading"]};font-size:22px;'
        f'margin:28px 0 18px;">Download Links</h3>'
    )
    for h, dl_url, size, *_ in outputs:
        parts.append(
            f'<h4 style="{head}">{h}p x264 {fps_txt}fps '
            f'[{human(size)}]</h4>'
        )
        parts.append(
            f'<a class="mv-dl" data-url="{e(dl_url, quote=True)}" '
            f'href="{e(dl_url, quote=True)}" rel="noopener" style="{btn}">'
            '&#11015;&#9889; DOWNLOAD NOW &#9889;&#11015;</a>'
        )

    # 7) Full-movie synopsis/plot — intentionally AFTER all download buttons.
    # The AI description is the site's full-movie, spoiler-light synopsis.
    # Keep the heading visible even if a future metadata provider returns no text.
    synopsis = [e(p) for p in (meta.get("synopsis") or meta.get("description") or []) if str(p).strip()]
    if not synopsis:
        synopsis = [
            f"{title} is presented as a complete film story, following its central characters as they face the circumstances and conflict that shape the narrative.",
            "The story develops through the characters' choices, relationships and challenges, building toward the film's larger dramatic direction without revealing its ending."
        ]

    if meta.get("tagline"):
        parts.append(hr)
        parts.append(
            f'<p style="text-align:center;color:{colors["heading"]};font-size:20px;'
            f'font-weight:700;margin:22px 0 14px;"><i>{e(meta["tagline"])}</i></p>'
        )

    parts.append(
        f'<h3 style="text-align:center;color:{colors["heading"]};font-size:22px;'
        f'margin:28px 0 18px;">Movie Synopsis / Plot</h3>'
    )
    for p in synopsis[:2]:
        parts.append(f'<p style="line-height:1.75;font-size:18px;">{p}</p>')

    parts.append(hr)

    # Sync Movie Info -> Movie Name with the title that is actually visible
    # at the top of the Blogger post. This means the user can manually edit
    # the Blogger title and Movie Info will follow it automatically.
    # We strip the technical suffix after the first | so a title such as
    # "KGF Chapter 2 | 720p | Watch Online & Download" becomes "KGF Chapter 2".
    parts.append(
        '<script>(function(){'
        'function syncMovieName(){'
        'var value=document.querySelector(\'.mv-movie-name-value\');'
        'if(!value)return;'
        'var selectors=[\'h1.post-title\',\'h1.entry-title\',\'.post-title h1\',\'.entry-title\',\'h1[itemprop="name"]\',\'.post h1\',\'h1\'];'
        'var postTitle="";'
        'for(var i=0;i<selectors.length;i++){'
        'var els=document.querySelectorAll(selectors[i]);'
        'for(var j=0;j<els.length;j++){'
        'var t=(els[j].textContent||"").replace(/\\s+/g," ").trim();'
        'if(t && !/^(movie info|watch online|screenshots|download links|movie synopsis \\/ plot)$/i.test(t)){postTitle=t;break;}'
        '}'
        'if(postTitle)break;'
        '}'
        'if(!postTitle){return;}'
        'var name=postTitle.split("|")[0].trim();'
        'name=name.replace(/\\s*\\(\\d{4}\\)\\s*$/," ").trim();'
        'if(name){value.textContent=name;}'
        '}'
        'if(document.readyState===\'loading\'){document.addEventListener(\'DOMContentLoaded\',syncMovieName);}else{syncMovieName();}'
        'setTimeout(syncMovieName,500);'
        'setTimeout(syncMovieName,1500);'
        '})();</script>'
    )
    parts.append(TIMER_SCRIPT)
    return "\n".join(parts)

# ---------- main pipeline ----------
def process(video, processed_folder):
    name = video["name"]
    vid = video["id"]
    log(f"\n=== Processing: {name} ===")
    st = load_state(vid)
    if st:
        log("  Resuming from saved state. Finished so far:",
            [k for k in ("meta", "outputs", "post_url") if st.get(k)])

    job = WORK / vid
    job.mkdir(parents=True, exist_ok=True)
    src = str(job / "source.mp4")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", Path(name).stem).strip("-").lower() or "movie"

    # ---- everything already done except the Drive move? ----
    if not st.get("post_url"):
        need_time(30)
        log("Downloading original...")
        download(vid, src)
        dur, w, h, fps = probe(src)
        st.update(dur=dur, w=w, h=h, fps=fps)
        save_state(vid, st)
        short = min(w, h)
        log(f"Duration {dur / 60:.1f} min, {w}x{h}, {fps:.2f} fps")

        site_labels = get_site_labels()
        st.setdefault("ia_item", make_ia_item_id(slug))
        ia_item = st["ia_item"]
        log("Internet Archive item:", ia_item)

        # ---- Stage A1: extract movie info FIRST (title, description, labels, posters) ----
        if "meta" not in st:
            need_time(15)
            log("Extracting movie info first (quick frames + audio -> Gemini)...")
            frames, audio = quick_analysis_inputs(src, dur, job)
            meta = analyze(name, frames, audio, site_labels)
            log("  Title:", meta["title"])
            log("  Labels:", meta["labels"])
            st["meta"] = meta
            save_state(vid, st)
            log("  Checkpoint saved: metadata")
        meta = st["meta"]

        # ---- Stage A2: video work (screenshots, poster, player thumbnail) ----
        if not (st.get("thumb_url") and st.get("shot_urls")):
            need_time(40)
            log("Making AI-selected screenshots...")
            shots = make_screenshots(src, dur, w, h, job)

            log("Making poster thumbnail (TMDB/IMDb first, then Google, then source frame)...")
            thumb = make_thumbnail(
                src, dur, w, h, job,
                movie_title=meta["title"],
                filename_hint=name,
                poster_urls=meta.get("poster_urls"),
            )

            # Image SEO: descriptive filenames (interstellar-2014-poster.webp)
            seo_base = seo_image_base(meta, slug)
            thumb = seo_rename(thumb, f"{seo_base}-poster")
            shots = [seo_rename(p, f"{seo_base}-screenshot-{i}")
                     for i, p in enumerate(shots, 1)]
            log("  Image filenames:", Path(thumb).name, "+", len(shots), "screenshots")

            log("Uploading images (WebP) to Internet Archive...")
            st["thumb_url"] = ia_upload_file(thumb, ia_item, meta["title"])
            st["shot_urls"] = [ia_upload_file(p, ia_item, meta["title"]) for p in shots]
            log("Making player thumbnail from the first video frame...")
            pp = seo_rename(make_player_poster(src, dur, job), f"{seo_base}-player-poster")
            st["player_poster_url"] = ia_upload_file(pp, ia_item, meta["title"])
            save_state(vid, st)
            log("  Checkpoint saved: images")
        if not st.get("player_poster_url"):
            log("Making player thumbnail from the first video frame...")
            pp = make_player_poster(src, dur, job)
            pp = seo_rename(pp, f"{seo_image_base(meta, slug)}-player-poster")
            st["player_poster_url"] = ia_upload_file(pp, st["ia_item"], meta["title"])
            save_state(vid, st)

        # ---- Stage B: each resolution is its own checkpoint ----
        # Widescreen films (e.g. 1920x816) are "1080p" by WIDTH, so a resolution
        # is offered when either its short side or its long side fits the source.
        def fits(t):
            return (t <= short * 1.05
                    or (w >= h and LONG_SIDE.get(t, round(t * 16 / 9)) <= w * 1.05))
        play_t = sorted({t for t in PLAY_RESOLUTIONS if fits(t)})
        dl_t = sorted({t for t in DOWNLOAD_RESOLUTIONS if fits(t)})
        if not play_t:
            play_t = [short]
        if not dl_t:                       # small source: best available is the download
            dl_t = [max(play_t)]
        targets = sorted(set(play_t) | set(dl_t))
        log("  Watch qualities:", play_t, "| Download qualities:", dl_t)
        done = {int(o[0]): o for o in st.get("outputs", [])}
        partial = st.setdefault("partial", {})
        for t in targets:
            if t in done:
                log(f"  {t}p already uploaded; skipping.")
                continue
            need_play, need_dl = t in play_t, t in dl_t
            need_time(120 if t >= 1440 else 60)
            part = partial.setdefault(str(t), {})
            mp4 = str(job / f"{slug}_{t}p.mp4")

            def ensure_mp4():
                if os.path.exists(mp4):
                    return
                if part.get("play_url"):
                    try:
                        log(f"  Restoring {t}p MP4 from Internet Archive for repacking...")
                        fetch_file(part["play_url"], mp4)
                        return
                    except Exception as ex:  # noqa
                        log(f"  Could not restore MP4 ({ex}); converting again...")
                log(f"Converting to {t}p...")
                transcode(src, t, w, h, mp4)

            # 1) MP4 for watching (its own checkpoint)
            if need_play and not part.get("play_url"):
                log(f"Converting to {t}p (MP4 for the player)...")
                transcode(src, t, w, h, mp4)
                part["play_size"] = os.path.getsize(mp4)
                log(f"Uploading {t}p MP4 to Internet Archive ({human(part['play_size'])})...")
                part["play_url"] = ia_upload_file(mp4, ia_item, meta["title"])
                save_state(vid, st)
                log(f"  Checkpoint saved: {t}p MP4 (watch)")

            # 2) file for the Download button (mkv by default, repacked from the MP4)
            dl_url, size = "", 0
            if need_dl:
                if IA_FILE_EXT == "mp4" and need_play:
                    dl_url, size = part["play_url"], int(part["play_size"])
                elif IA_FILE_EXT == "mp4":
                    ensure_mp4()
                    size = os.path.getsize(mp4)
                    log(f"Uploading {t}p MP4 (download) to Internet Archive ({human(size)})...")
                    dl_url = ia_upload_file(mp4, ia_item, meta["title"])
                else:
                    dl_file = str(job / f"{slug}_{t}p.{IA_FILE_EXT}")
                    ensure_mp4()
                    log(f"Repacking {t}p to {IA_FILE_EXT.upper()} (no re-encode)...")
                    remux_copy(mp4, dl_file)
                    size = os.path.getsize(dl_file)
                    log(f"Uploading {t}p {IA_FILE_EXT.upper()} to Internet Archive ({human(size)})...")
                    dl_url = ia_upload_file(dl_file, ia_item, meta["title"])
                    if os.path.exists(dl_file):
                        os.remove(dl_file)

            st.setdefault("outputs", []).append(
                [t, dl_url, size, part.get("play_url", "")])
            partial.pop(str(t), None)
            save_state(vid, st)
            log(f"  Checkpoint saved: {t}p "
                f"({'watch' if need_play else ''}{' + ' if need_play and need_dl else ''}"
                f"{'download' if need_dl else ''})")
            if os.path.exists(mp4):
                os.remove(mp4)  # free disk
        all_out = sorted(
            ((int(o[0]), o[1], int(o[2]), (o[3] if len(o) > 3 else o[1])) for o in st["outputs"]),
            key=lambda x: x[0])
        play_outputs = [o for o in all_out if o[3] and o[0] in play_t]
        outputs = [o for o in all_out if o[1] and o[0] in dl_t]   # download buttons

        # ---- Stage C: Blogger post ----
        labels = list(meta["labels"])
        if not labels:
            unc = next((l for l in site_labels if l.lower() == "uncategorized"), None)
            labels = [unc] if unc else [g for g in meta["genres"]][:2] + [meta["language"]]
        labels = [str(l)[:40] for l in labels if l][:8]

        log("Shortening download links with ShrtFly...")
        html_outputs = shorten_outputs(outputs, st, vid)
        content = build_html(meta, st["thumb_url"], st["shot_urls"], html_outputs, fps, dur,
                              st.get("player_poster_url", ""), play_outputs)
        final_post_title = make_final_post_title(meta, outputs)
        log("  Final Blogger title:", final_post_title)
        body = {"kind": "blogger#post",
                "title": final_post_title,
                "content": content, "labels": labels}
        post = retry(lambda: blogger.posts().insert(
            blogId=BLOG_ID, body=body, isDraft=not PUBLISH).execute())
        st["post_url"] = post.get("url") or post.get("id") or "created"
        save_state(vid, st)   # never create the same post twice
        log("Blogger post created:", st["post_url"],
            "(DRAFT)" if not PUBLISH else "(PUBLISHED)")

    # The source video is touched ONLY after the Blogger post exists.
    if not st.get("post_url"):
        raise RuntimeError("Blogger post not created yet - keeping the Drive video.")
    record_processed(video)
    move_to_processed(vid, processed_folder)
    clear_state(vid)
    shutil.rmtree(job, ignore_errors=True)


def main():
    WORK.mkdir(exist_ok=True)
    videos = list_videos()
    if not videos:
        log("No new videos in the input folder. Nothing to do.")
        return 0
    processed = None if ACCOUNTS_JSON else ensure_folder("_processed")
    failed = 0
    taken = 0
    for v in videos:
        if taken >= MAX_VIDEOS:
            break
        if is_duplicate(v):
            log(f"  '{v.get('name')}' was already processed before - duplicate.")
            try:
                move_duplicate(v, processed)
            except Exception as ex:  # noqa
                log(f"  Could not move duplicate ({str(ex)[:200]}).")
            continue
        if not try_claim(v):
            continue
        taken += 1
        if DEDUP:
            _load_processed_fingerprints().update(_fp_keys(v))
        try:
            process(v, processed)
        except ResumeLater as ex:
            log(f"\n>>> Time almost up ({ex}). Progress saved; the workflow will continue.")
            return RESUME_EXIT_CODE
        except Exception:  # noqa
            failed += 1
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
