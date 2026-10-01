"""HTTP smoke checks and optional upload; saves reproducible evidence."""
import argparse
import json
import time
import urllib.request
import urllib.error

from .config import STATE, DEFAULT_DATA


def request(base, path, body=None):
    raw = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=raw,
                                 headers={"Content-Type": "application/json"} if raw else {})
    start = time.perf_counter()
    try:
        response = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        payload = json.loads(response.read())
        return {"status": response.status, "seconds": time.perf_counter()-start, "body": payload}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--label", default="http")
    args = parser.parse_args()
    evidence = {"health_before": request(args.url, "/health")}
    if args.upload:
        import requests
        with DEFAULT_DATA.open("rb") as f:
            r = requests.post(args.url + "/data/upload", files={"file": f}, timeout=120)
        evidence["upload"] = {"status": r.status_code, "body": r.json()}
        assert r.status_code == 200, r.text
        assert r.json()["rows"] == 263040
    example = request(args.url, "/data/example")
    assert example["status"] == 200, example
    body = example["body"]
    evidence["first_predict"] = request(args.url, "/predict", body)
    evidence["second_predict"] = request(args.url, "/predict", body)
    assert evidence["first_predict"]["status"] == evidence["second_predict"]["status"] == 200
    body["sequence"].pop()
    evidence["invalid_19_points"] = request(args.url, "/predict", body)
    assert evidence["invalid_19_points"]["status"] == 422
    evidence["health_after"] = request(args.url, "/health")
    STATE.mkdir(parents=True, exist_ok=True)
    output = STATE / f"verification_{args.label}.json"
    output.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    print(f"Evidence: {output}")


if __name__ == "__main__":
    main()
