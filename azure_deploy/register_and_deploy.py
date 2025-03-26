from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

ws = Workspace.from_config()

# Register model
model = Model.register(
    workspace=ws,
    model_path="xgb_model.pkl",
    model_name="xgb_model"
)

# Define environment
env = Environment.from_conda_specification(name="xgb-env", file_path="env.yml")

# Inference config
inference_config = InferenceConfig(entry_script="score.py", environment=env)

# Deployment config
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(
    workspace=ws,
    name="xgb-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")
