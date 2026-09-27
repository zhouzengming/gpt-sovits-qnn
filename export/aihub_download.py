"""Step 5: download the compiled context binaries from AI Hub -> <out>/<graph>.bin.

`hub.get_model(id).download()` fetches 256 MiB ranges without a read timeout, which can hang forever behind a
proxy. The same S3 object is fetched here in 16 MiB ranges with per-range deadlines, retries and resume (through
qai_hub's internal credential helper; --simple uses the public API instead).
Adapted from github.com/zhouzengming/qwen3-reranker-qnn (Apache-2.0).

  python export/aihub_download.py --state work/myvoice/aihub_state.json --out work/myvoice/bin
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CHUNK = 16 * 1024 * 1024
WORKERS = 4
CHUNK_DEADLINE_S = 120  # per 16 MiB range; slower ranges are aborted and retried


class RangeFetcher:
    def __init__(self, model_id):
        self.model_id = model_id
        self.lock = threading.Lock()
        self.refresh()

    def refresh(self):
        import botocore.session
        from botocore.client import Config
        from qai_hub import public_rest_api as api

        c = api._get_model_download_response(hub.hub._global_client.config, self.model_id).credentials
        client = botocore.session.get_session().create_client(
            "s3", region_name=c.region or None, aws_access_key_id=c.access_key_id,
            aws_secret_access_key=c.secret_access_key, aws_session_token=c.session_token,
            config=Config(signature_version="s3v4", s3={"use_accelerate_endpoint": c.use_transfer_acceleration},
                          connect_timeout=15, read_timeout=30, retries={"max_attempts": 3}))
        with self.lock:
            self.creds, self.client = c, client

    def size(self):
        return self.client.head_object(Bucket=self.creds.bucket, Key=self.creds.key)["ContentLength"]

    def get(self, start, end):
        for attempt in range(1, 30):
            try:
                with self.lock:
                    client, c = self.client, self.creds
                body = client.get_object(Bucket=c.bucket, Key=c.key, Range=f"bytes={start}-{end}")["Body"]
                # read_timeout only bounds the gap between packets; a connection that trickles a few bytes
                # now and then never trips it, so also enforce a deadline for the whole range
                deadline = time.monotonic() + CHUNK_DEADLINE_S
                parts = []
                for piece in body.iter_chunks(1024 * 1024):
                    parts.append(piece)
                    if time.monotonic() > deadline:
                        body.close()
                        raise TimeoutError(f"range not finished within {CHUNK_DEADLINE_S}s")
                data = b"".join(parts)
                if len(data) == end - start + 1:
                    return data
                raise IOError(f"short read {len(data)}")
            except Exception as e:  # noqa: BLE001
                print(f"  retry range {start}-{end} (attempt {attempt}): {type(e).__name__}: {str(e)[:120]}", flush=True)
                if any(s in str(e) for s in ("ExpiredToken", "AccessDenied", "403")):
                    self.refresh()  # temporary credentials expired
                time.sleep(min(2 * attempt, 20))
        raise RuntimeError(f"range {start}-{end} failed repeatedly")


def download_ranged(model_id, dest):
    f = RangeFetcher(model_id)
    size = f.size()
    part = dest.with_suffix(dest.suffix + ".part")
    done_file = dest.with_suffix(dest.suffix + ".done.json")
    done = set(json.loads(done_file.read_text())) if done_file.exists() and part.exists() else set()
    if not part.exists():
        with open(part, "wb") as fh:
            fh.truncate(size)
    chunks = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    lock = threading.Lock()

    def work(c):
        data = f.get(*c)
        with lock:
            with open(part, "r+b") as fh:
                fh.seek(c[0])
                fh.write(data)
            done.add(c[0])
            done_file.write_text(json.dumps(sorted(done)))

    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(work, [c for c in chunks if c[0] not in done]), 1):
            if i % 8 == 0:
                print(f"  {dest.name}: {len(done)}/{len(chunks)} chunks", flush=True)
    part.rename(dest)
    done_file.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="state file of aihub_compile.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--simple", action="store_true", help="use hub.get_model().download()")
    args = ap.parse_args()
    state = json.loads(Path(args.state).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for g, stem in C.GRAPHS.items():
        model_id = state.get(g, {}).get("ctx_model")
        if not model_id:
            raise SystemExit(f"no context binary for {g} in {args.state}; run aihub_compile.py first")
        dest = out / f"{stem}.bin"
        if dest.exists():
            print(f"[download] skip {dest.name}")
            continue
        if args.simple:
            hub.get_model(model_id).download(str(dest))
        else:
            download_ranged(model_id, dest)
        print(f"[download] {dest.name} {dest.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
