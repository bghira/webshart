import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading

from PIL import Image
import pytest

import webshart
from webshart.whitespace import measure


def stats(image):
    return measure(image, white_threshold=242, black_threshold=13)


def test_neutral_fractions_and_contiguous_border():
    im = Image.new("RGB", (100, 100), (242, 242, 242))
    im.paste((13, 13, 13), (15, 12, 82, 86))
    im.putpixel((0, 50), (255, 241, 255))  # A one-pixel mark protects the left edge.
    im.putpixel((50, 50), (0, 14, 0))
    fields = stats(im)
    assert fields["whitespace_crop"] == [0, 8, 86, 90]
    assert fields["white_fraction"] == (10000 - 67 * 74 - 1) / 10000
    assert fields["black_fraction"] == (67 * 74 - 1) / 10000


@pytest.mark.parametrize(
    ("color", "white", "black"),
    [
        ("white", 1, 0),
        ("black", 0, 1),
        ("magenta", 0, 0),
        ("gray", 0, 0),
    ],
)
def test_uniform_images(color, white, black):
    assert stats(Image.new("RGB", (20, 20), color)) == {
        "white_fraction": white,
        "black_fraction": black,
    }


def test_white_background_with_tiny_subject_and_transparency():
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    im.putpixel((70, 60), (0, 0, 0, 255))
    assert stats(im) == {
        "white_fraction": 0.9999,
        "black_fraction": 0.0001,
        "whitespace_crop": [15, 15, 85, 85],
    }


def test_exif_orients_crop_coordinates():
    im = Image.new("RGB", (100, 60), "white")
    im.paste("red", (14, 0, 100, 60))
    im.getexif()[274] = 6
    assert stats(im)["whitespace_crop"] == [0, 12, 60, 100]


def fixture(root):
    root.mkdir(parents=True)
    image = Image.new("RGB", (100, 100), "white")
    image.paste("blue", (14, 0, 100, 100))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    files = {}
    with tarfile.open(root / "000.tar", "w") as tar:
        for name, data in (
            ("test.png", buf.getvalue()),
            ("test.txt", b"caption"),
            ("broken.jpg", b"bad"),
        ):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    with tarfile.open(root / "000.tar") as tar:
        for entry in tar:
            files[entry.name] = {"offset": entry.offset_data, "length": entry.size}
    files["test.png"].update(
        width=100, height=100, json_label={"text": "caption", "bbox": [0, 1, 2, 3]}
    )
    files["broken.jpg"]["whitespace_crop"] = [1, 1, 2, 2]
    index = {
        "path": "000.tar",
        "filesize": (root / "000.tar").stat().st_size,
        "custom": "keep",
        "files": files,
    }
    (root / "000.json").write_text(json.dumps(index))
    return index


def cli(*args, env=None):
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from webshart import main; main()",
            "whitespace-crop",
            *map(str, args),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_cli_enriched_indexes_and_loader(tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    original = fixture(source)
    run = cli("--source", source, "--destination", output, "--workers", 2)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {
        "shards": 1,
        "scanned": 1,
        "cropped": 1,
        "unreadable": 1,
    }
    enriched = json.loads((output / "000.json").read_text())
    image = enriched["files"]["test.png"]
    assert image["white_fraction"] == 0.14 and image["whitespace_crop"] == [
        10,
        0,
        100,
        100,
    ]
    assert "whitespace_crop" not in enriched["files"]["broken.jpg"]
    assert original["files"]["test.txt"] == enriched["files"]["test.txt"]
    assert enriched["custom"] == "keep"
    assert json.loads((source / "000.json").read_text()) == original
    dataset = webshart.discover_dataset(str(source), metadata=str(output))
    row = webshart.TarDataLoader(dataset, load_file_data=False).get_metadata(0)[
        "test.png"
    ]
    assert (
        row["white_fraction"] == 0.14
        and row["json_label"] == original["files"]["test.png"]["json_label"]
    )


def test_atomic_on_mismatched_index_and_list_format(tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    index = fixture(source)
    index["files"] = [dict(filename=k, **v) for k, v in index["files"].items()]
    (source / "000.json").write_text(json.dumps(index))
    args = ("--source", source, "--destination", output)
    assert cli(*args).returncode == 0
    before = (output / "000.json").read_bytes()
    assert json.loads(before)["files"][0]["black_fraction"] == 0
    index["files"][0]["offset"] += 1
    (source / "000.json").write_text(json.dumps(index))
    assert cli(*args).returncode != 0
    assert (output / "000.json").read_bytes() == before


def test_remote_cli_with_separate_index(tmp_path):
    root = tmp_path / "fixture"
    fixture(root)
    rev = "a" * 40
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def serve(self, body):
            requests.append((self.command, self.path))
            repo = "test/index" if "test/index" in self.path else "test/images"
            filename = "000.json" if repo == "test/index" else "000.tar"
            if self.path.startswith("/api/datasets/"):
                data = json.dumps(
                    {"id": repo, "sha": rev, "siblings": [{"rfilename": filename}]}
                ).encode()
            elif self.path == f"/datasets/{repo}/resolve/{rev}/{filename}":
                data = (root / filename).read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                (
                    "application/json"
                    if self.path.startswith("/api/")
                    else "application/octet-stream"
                ),
            )
            self.send_header("X-Repo-Commit", rev)
            self.send_header("ETag", f'"{hashlib.sha256(data).hexdigest()}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if body:
                self.wfile.write(data)

        def do_GET(self):
            self.serve(True)

        def do_HEAD(self):
            self.serve(False)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        env = {
            **os.environ,
            "HF_ENDPOINT": f"http://127.0.0.1:{server.server_port}",
            "HF_HOME": str(tmp_path / "hub"),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_OFFLINE": "0",
        }
        env.pop("HF_TOKEN", None)
        result = cli(
            "--source",
            "test/images",
            "--metadata",
            "test/index",
            "--destination",
            tmp_path / "out",
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert (
            requests.count(("GET", f"/datasets/test/images/resolve/{rev}/000.tar")) == 1
        )
        assert json.loads((tmp_path / "out/000.json").read_text())["files"]["test.png"][
            "whitespace_crop"
        ] == [10, 0, 100, 100]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_tight_crop_keeps_interior_whitespace_and_ignores_black_borders():
    im = Image.new("RGB", (100, 100), "white")
    im.putpixel((40, 40), (0, 255, 0))
    im.putpixel((50, 50), (0, 0, 0))
    fields = measure(im, white_threshold=255, black_threshold=0, tight=True)
    assert fields["whitespace_crop"] == [40, 40, 51, 51]
    assert fields["black_fraction"] == 1 / 10000
    im = Image.new("RGB", (100, 100), "black")
    im.paste("white", (20, 20, 80, 80))
    assert "whitespace_crop" not in measure(
        im, white_threshold=242, black_threshold=13, tight=True
    )


def test_upload_sends_only_selected_indexes(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import huggingface_hub

    source, output = tmp_path / "source", tmp_path / "out"
    fixture(source / "a")
    fixture(source / "b")
    output.mkdir()
    (output / "unrelated.json").write_text("{}")
    calls = []

    class Hub:
        def __init__(self, token):
            pass

        def dataset_info(self, repo, revision):
            return SimpleNamespace(sha="initial-head")

        def create_commit(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "webshart",
            "whitespace-crop",
            "--source",
            str(source),
            "--destination",
            str(output),
            "--range",
            "1,2",
            "--push-to-hub",
            "test/index",
            "--path-in-repo",
            "scan",
        ],
    )
    webshart.main()
    assert calls[0]["parent_commit"] == "initial-head"
    assert [op.path_in_repo for op in calls[0]["operations"]] == ["scan/b/000.json"]
