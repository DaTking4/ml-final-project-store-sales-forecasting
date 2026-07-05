import mlflow
import os


def init_tracking():
    try:
        import dagshub
    except ImportError as exc:
        raise ImportError(
            "DagsHub tracking is enabled, but the 'dagshub' package is not installed. "
            "Install it with: python -m pip install dagshub"
        ) from exc

    repo_owner = os.environ.get("DAGSHUB_REPO_OWNER", "dkhak22")
    repo_name = os.environ.get(
        "DAGSHUB_REPO_NAME",
        "ml-final-project-store-sales-forecasting",
    )
    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

    dagshub.init(
        repo_owner=repo_owner,
        repo_name=repo_name,
        mlflow=True,
    )
    mlflow.set_tracking_uri(tracking_uri)

    print("MLflow tracking URI:", mlflow.get_tracking_uri())
