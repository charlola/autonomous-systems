import mlflow

def mlflow_logging(params, metrics, experiment):
    """experiment should be specified in args.experiment"""

    mlflow.set_tracking_uri("http://159.65.120.229:5000")
    mlflow.set_experiment(experiment)

    mlflow.log_metrics(metrics)
    mlflow.log_params(params)