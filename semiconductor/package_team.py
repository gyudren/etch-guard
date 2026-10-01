"""Create a clean ZIP with explicit allowlist; no virtualenv, .env or MLflow DB."""
import hashlib
import json
import zipfile
from .config import ROOT, STATE, DEFAULT_DATA


def main():
    output = ROOT / "team_share"
    output.mkdir(exist_ok=True)
    files = [ROOT / "requirements.txt", ROOT / "TEAM_DAY2.md", ROOT / "TEAM_VALIDATION.md", ROOT / "compose.semiconductor.yml",
             ROOT / ".dockerignore", DEFAULT_DATA,
             ROOT / "scripts/generate_semiconductor_etch_data.py"]
    files += sorted(p for p in (ROOT / "semiconductor").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    pointer = json.loads((STATE / "local.json").read_text())
    bundle = STATE / "bundles" / pointer["bundle"]
    files += [STATE / "local.json"] + [bundle / n for n in ("model.keras", "scaler.json", "metrics.json", "example.json")]
    files += sorted(STATE.glob("verification_*.json"))
    dest = output / "EtchGuard_Day2.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        manifest = {}
        for p in files:
            relative = p.relative_to(ROOT).as_posix()
            z.write(p, f"EtchGuard_Day2/{relative}")
            manifest[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
        z.writestr("EtchGuard_Day2/MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
    print(dest)
    print(f"{len(files)} files, {dest.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
