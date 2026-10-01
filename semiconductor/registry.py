"""Model Registry 운영 CLI: 버전 이력 조회와 수동 롤백(Production 별칭 되돌리기).

python -m semiconductor.registry list
python -m semiconductor.registry rollback --version 1   # 이후 서버 재시작(eager)으로 반영
"""
import argparse

from .config import MODEL_NAME, ALIAS, tracking_uri


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--version", required=True)
    args = parser.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    current = client.get_registered_model(MODEL_NAME).aliases.get(ALIAS)
    if args.command == "list":
        for v in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version)):
            run = client.get_run(v.run_id)
            mark = "  ← Production" if str(v.version) == str(current) else ""
            print(f"v{v.version:<3} mode={v.tags.get('mode', '?'):<9} gate_passed={v.tags.get('gate_passed'):<5} "
                  f"rmse={run.data.metrics.get('rmse', float('nan')):.3f}{mark}")
        return
    target = client.get_model_version(MODEL_NAME, args.version)
    if target.tags.get("gate_passed") != "True":
        raise SystemExit(f"v{args.version} did not pass the deployment gate; refusing to roll back to it")
    client.set_registered_model_alias(MODEL_NAME, ALIAS, args.version)
    print(f"[ROLLBACK] {ALIAS}: v{current} → v{args.version}. Restart the server to serve it.")


if __name__ == "__main__":
    main()
