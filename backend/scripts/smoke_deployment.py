"""Verify an HTTP deployment including the bundled frontend and Neural export."""

import argparse
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request


def request(opener, url, data=None, headers=None):
    message = urllib.request.Request(url, data=data, headers=headers or {})
    with opener.open(message, timeout=20) as response:
        return response.read(), response.headers.get("Content-Type", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    origin = args.url.rstrip("/")
    if urllib.parse.urlsplit(origin).hostname in {"localhost", "127.0.0.1", "::1"}:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        opener = urllib.request.build_opener()
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            raw, _ = request(opener, origin + "/api/ready")
            ready = json.loads(raw)
            if ready.get("status") == "ready":
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Deployment did not become model-ready")
        time.sleep(2)
    frontend, content_type = request(opener, origin + "/")
    if "text/html" not in content_type or b'id="root"' not in frontend:
        raise ValueError("Bundled frontend is missing")
    info = json.loads(request(opener, origin + "/api/model/info")[0])
    if not info.get("available") or info["version"] != ready["model_version"]:
        raise ValueError("Ready endpoint and loaded model disagree")
    demo = json.loads(request(opener, origin + "/api/demo/0")[0])
    boundary = "photonica-deployment-smoke"
    body = bytearray()
    for image in demo["images"]:
        body.extend(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="images"; '
                'filename="target.png"\r\nContent-Type: image/png\r\n\r\n'
            ).encode()
        )
        body.extend(base64.b64decode(image, validate=True))
        body.extend(b"\r\n")
    body.extend(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="method"\r\n\r\n'
            f"neural\r\n--{boundary}--\r\n"
        ).encode()
    )
    raw, _ = request(
        opener,
        origin + "/api/synthesize",
        bytes(body),
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    result = json.loads(raw)
    if (
        result["iterations"] != 0
        or result["method"] != "neural"
        or result["model_version"] != info["version"]
        or len(result["reconstruction_pngs"]) != 4
    ):
        raise ValueError("Neural inference/export contract failed")
    for encoded in [result["phase_png"], *result["reconstruction_pngs"]]:
        png = base64.b64decode(encoded, validate=True)
        if (
            png[:8] != b"\x89PNG\r\n\x1a\n"
            or png[16:24] != b"\x00\x00\x00\x80" * 2
            or png[24:26] != b"\x08\x00"
        ):
            raise ValueError("Expected 128x128 eight-bit grayscale PNG export")
    print(
        json.dumps(
            {
                "url": origin,
                "model_version": info["version"],
                "frontend": "ok",
                "neural_iterations": result["iterations"],
                "reconstructions": 4,
            }
        )
    )


if __name__ == "__main__":
    main()
