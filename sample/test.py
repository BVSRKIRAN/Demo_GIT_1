import mlflow
from random import random, randint
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlflow import MlflowClient


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

EXPERIMENT_NAME = "mlflow-demo1"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
client = MlflowClient()

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

for idx, depth in enumerate([1, 2, 5, 10, 20]):
    mlflow.autolog(log_models=True, exclusive=False)
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Start MLflow
    RUN_NAME = f"run_{idx}"
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME,nested=True) as run:
        # Retrieve run id
        RUN_ID = run.info.run_id

    #    # Track parameters
    #    mlflow.log_param("depth", depth)

    #    # Track metrics
    #   mlflow.log_metric("accuracy", accuracy)

    #    # Track model
    #    mlflow.sklearn.log_model(clf, "classifier")
        
        # Log the sklearn model and register as version 1
    #mlflow.sklearn.log_model(
    #    sk_model=clf,
    #   artifact_path="sklearn-model",
    #    registered_model_name="sk-learn-DT-model",
    #)

# fetch the auto logged parameters and metrics for ended run
print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
