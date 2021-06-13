import mlflow

def log_artifact_example():
    """artifacts are files/plots and saved in server home dir"""
    with open("output.txt", "w") as f:
        f.write("Thank Charlotte later")
    mlflow.log_artifact("output.txt")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test")

params = {"p1": "A", "p2": "B"}
metric = {"m1": 12, "m2": 34}


mlflow.log_metrics(metric)
mlflow.log_params(params)




