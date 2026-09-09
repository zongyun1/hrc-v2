#!/usr/bin/env python3
"""Download the public OneDrive assets folder via SharePoint's drive API.

This is a fallback for machines without an rclone OneDrive remote. It uses the
anonymous SharePoint page token exposed by the public folder link, walks the
drive tree recursively, and downloads files with restart-safe ``.part`` files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.cookiejar
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_SHARE_URL = (
    "https://umass-my.sharepoint.com/:f:/g/personal/"
    "qinhongzhou_umass_edu/"
    "IgAv1Rf-uN_AQruMkgNuIfhZAQuAIwcvN5yFBne8wFNUXGY?e=31tltm&download=1"
)


class SharePointClient:
    def __init__(self, share_url: str):
        self.share_url = share_url
        self._lock = threading.Lock()
        self._opener = urllib.request.build_opener()
        self.drive_url = ""
        self.token = ""
        self.root_item_id = ""
        self.root_server_path = ""
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            cookie_jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(cookie_jar)
            )
            req = urllib.request.Request(
                self.share_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with opener.open(req, timeout=120) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            token = self._extract(
                html,
                r'"\.driveAccessToken"\s*:\s*"access_token=([^"]+)"',
                "driveAccessToken",
            )
            drive_url = self._extract(
                html,
                r'"\.driveUrl"\s*:\s*"([^"]+)"',
                "driveUrl",
            )
            current_item_url = self._extract(
                html,
                r'"CurrentFolderSpItemUrl"\s*:\s*"([^"]+)"',
                "CurrentFolderSpItemUrl",
            )
            root_server_path = self._extract(
                html,
                r'"rootFolder"\s*:\s*"([^"]+)"',
                "rootFolder",
            )

            self.token = token
            self.drive_url = self._decode_jsonish_url(drive_url).rstrip("/")
            self._opener = opener
            current_item_url = self._decode_jsonish_url(current_item_url)
            match = re.search(r"/items/([^?]+)", current_item_url)
            if not match:
                raise RuntimeError("Could not extract root folder item id")
            self.root_item_id = match.group(1)
            self.root_server_path = self._decode_jsonish_url(root_server_path)

    @staticmethod
    def _extract(text: str, pattern: str, label: str) -> str:
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"Could not extract {label} from SharePoint page")
        return match.group(1)

    @staticmethod
    def _decode_jsonish_url(value: str) -> str:
        return value.replace(r"\u002f", "/").replace(r"\/", "/")

    def api_json(self, url: str, *, retries: int = 3) -> dict:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
        )
        try:
            with self._opener.open(req, timeout=180) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if retries > 0 and exc.code in {401, 403, 429, 503}:
                time.sleep(2)
                self.refresh()
                return self.api_json(url, retries=retries - 1)
            raise

    def item_children(self, item_id: str):
        params = urllib.parse.urlencode(
            {
                "$select": "name,id,size,folder,file,@content.downloadUrl",
                "$top": "200",
            }
        )
        url = f"{self.drive_url}/items/{item_id}/children?{params}"
        while url:
            data = self.api_json(url)
            yield from data.get("value", [])
            url = data.get("@odata.nextLink")

    def item_metadata(self, item_id: str) -> dict:
        params = urllib.parse.urlencode(
            {"$select": "name,id,size,file,@content.downloadUrl"}
        )
        return self.api_json(f"{self.drive_url}/items/{item_id}?{params}")

    def classic_json(self, url: str, *, retries: int = 3) -> dict:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json;odata=nometadata",
                "User-Agent": "Mozilla/5.0",
            },
        )
        try:
            with self._opener.open(req, timeout=180) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if retries > 0 and exc.code in {401, 403, 429, 503}:
                time.sleep(2)
                self.refresh()
                return self.classic_json(url, retries=retries - 1)
            raise

    @staticmethod
    def _classic_folder_url(server_rel_path: str, suffix: str) -> str:
        quoted = urllib.parse.quote(server_rel_path.replace("'", "''"), safe="/")
        return (
            "https://umass-my.sharepoint.com/personal/qinhongzhou_umass_edu"
            f"/_api/web/GetFolderByServerRelativeUrl('{quoted}')/{suffix}"
        )

    @staticmethod
    def classic_file_value_url(server_rel_path: str) -> str:
        quoted = urllib.parse.quote(server_rel_path.replace("'", "''"), safe="/")
        return (
            "https://umass-my.sharepoint.com/personal/qinhongzhou_umass_edu"
            f"/_api/web/GetFileByServerRelativeUrl('{quoted}')/$value"
        )

    def classic_children(self, server_rel_path: str):
        for folder in self._classic_values(self._classic_folder_url(server_rel_path, "Folders")):
            name = folder["Name"]
            if name == "Forms":
                continue
            yield {
                "name": name,
                "folder": {"childCount": int(folder.get("ItemCount") or 0)},
                "server_rel": folder["ServerRelativeUrl"],
                "classic": True,
            }
        for file_item in self._classic_values(self._classic_folder_url(server_rel_path, "Files")):
            yield {
                "name": file_item["Name"],
                "file": {},
                "size": int(file_item.get("Length") or 0),
                "server_rel": file_item["ServerRelativeUrl"],
                "unique_id": file_item.get("UniqueId"),
                "classic": True,
            }

    def _classic_values(self, url: str):
        while url:
            data = self.classic_json(url)
            yield from data.get("value", [])
            url = data.get("@odata.nextLink") or data.get("odata.nextLink")


def walk_tree(
    client: SharePointClient,
    item_id: str | None,
    rel_dir: Path,
    server_rel_path: str,
):
    try:
        iterator = client.item_children(item_id) if item_id else client.classic_children(server_rel_path)
        for item in iterator:
            rel = rel_dir / item["name"]
            if "folder" in item:
                child_server_rel = item.get("server_rel") or f"{server_rel_path}/{item['name']}"
                yield from walk_tree(client, item.get("id"), rel, child_server_rel)
            elif "file" in item:
                yield {
                    "id": item.get("id"),
                    "name": item["name"],
                    "rel": rel.as_posix(),
                    "size": int(item.get("size") or 0),
                    "url": item.get("@content.downloadUrl"),
                    "server_rel": item.get("server_rel"),
                    "unique_id": item.get("unique_id"),
                }
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403} and item_id:
            print(
                f"fallback classic listing for {rel_dir.as_posix() or '.'} "
                f"after HTTP {exc.code}",
                flush=True,
            )
            yield from walk_tree(client, None, rel_dir, server_rel_path)
            return
        raise RuntimeError(
            f"failed to list {rel_dir.as_posix() or '.'} "
            f"(item {item_id}, path {server_rel_path}): HTTP {exc.code} {exc.reason}"
        ) from exc


def download_one(client: SharePointClient, root: Path, file_info: dict) -> str:
    retryable_http = {401, 403, 429, 500, 502, 503, 504}
    last_exc: BaseException | None = None
    for attempt in range(6):
        try:
            return _download_one(client, root, file_info)
        except urllib.error.HTTPError as exc:
            if exc.code not in retryable_http:
                raise
            if exc.code in {401, 403}:
                client.refresh()
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_exc = exc

        if attempt < 5:
            time.sleep(min(60, 2 ** attempt))

    assert last_exc is not None
    raise last_exc


def _download_one(client: SharePointClient, root: Path, file_info: dict) -> str:
    rel = file_info["rel"]
    expected_size = int(file_info["size"])
    dest = root / rel
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and (expected_size == 0 or dest.stat().st_size == expected_size):
        return f"skip {rel}"
    if dest.exists() and expected_size and dest.stat().st_size != expected_size and not part.exists():
        dest.rename(part)

    start = part.stat().st_size if part.exists() else 0
    if expected_size and start > expected_size:
        part.unlink()
        start = 0

    metadata = file_info
    url = metadata.get("url")
    if not url:
        if file_info.get("id"):
            metadata = client.item_metadata(file_info["id"])
            url = metadata.get("@content.downloadUrl")
        elif file_info.get("server_rel"):
            url = client.classic_file_value_url(file_info["server_rel"])
    if not url:
        raise RuntimeError(f"No download URL for {rel}")

    mode = "ab" if start else "wb"
    use_classic_cookies = bool(file_info.get("server_rel") and not metadata.get("@content.downloadUrl"))
    headers = {"User-Agent": "Mozilla/5.0"}
    if not use_classic_cookies:
        headers["Authorization"] = f"Bearer {client.token}"
    if start:
        headers["Range"] = f"bytes={start}-"

    try:
        req = urllib.request.Request(url, headers=headers)
        with client._opener.open(req, timeout=600) as resp, part.open(mode) as fh:
            while True:
                chunk = resp.read(4 * 1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            client.refresh()
        raise

    actual_size = part.stat().st_size
    if expected_size and actual_size != expected_size:
        raise RuntimeError(f"incomplete {rel}: got {actual_size}, expected {expected_size}")
    part.replace(dest)
    return f"download {rel} ({actual_size} bytes)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share-url", default=DEFAULT_SHARE_URL)
    parser.add_argument("--output", default="assets")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="debug: max files")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = SharePointClient(args.share_url)

    print(f"drive: {client.drive_url}", flush=True)
    print(f"root item: {client.root_item_id}", flush=True)
    print(f"output: {output}", flush=True)
    print("listing files...", flush=True)

    files = []
    total_bytes = 0
    for idx, item in enumerate(
        walk_tree(client, client.root_item_id, Path(""), client.root_server_path),
        start=1,
    ):
        files.append(item)
        total_bytes += int(item["size"])
        if idx % 1000 == 0:
            print(f"  listed {idx} files ({total_bytes / 1024**3:.1f} GiB)", flush=True)
        if args.limit and len(files) >= args.limit:
            break

    print(f"listed {len(files)} files ({total_bytes / 1024**3:.1f} GiB)", flush=True)

    completed = 0
    skipped = 0
    downloaded = 0
    failures = []
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(download_one, client, output, item): item for item in files}
        for future in concurrent.futures.as_completed(future_map):
            item = future_map[future]
            completed += 1
            try:
                msg = future.result()
                if msg.startswith("skip "):
                    skipped += 1
                else:
                    downloaded += 1
            except Exception as exc:
                failures.append((item["rel"], str(exc)))
                msg = f"FAIL {item['rel']}: {exc}"

            if completed % 100 == 0 or msg.startswith("FAIL "):
                elapsed = max(1.0, time.time() - start_time)
                print(
                    f"{completed}/{len(files)} done, "
                    f"downloaded={downloaded}, skipped={skipped}, "
                    f"failed={len(failures)}, rate={completed / elapsed:.2f} files/s",
                    flush=True,
                )
                if msg.startswith("FAIL "):
                    print(msg, flush=True)

    if failures:
        print("failures:", flush=True)
        for rel, err in failures[:50]:
            print(f"  {rel}: {err}", flush=True)
        raise SystemExit(1)

    print(
        f"complete: files={len(files)}, downloaded={downloaded}, skipped={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
