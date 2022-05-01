# main.py

### Import packages and set up configurations
import os
from azureml.core import Workspace, Environment, Experiment, Dataset, ComputeTarget, Model
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.data import OutputFileDatasetConfig
from azureml.core.runconfig import RunConfiguration
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
ws = Workspace.from_config()
blob_store = ws.get_default_datastore()
compute_target = ComputeTarget(ws, "ml-ci")
myenv = Environment.get(workspace = ws, name = "ml_env")
runconfig = RunConfiguration()
runconfig.environment = myenv
experiment = Experiment(workspace = ws, name = "heart_disease_pipeline")
inf_config = InferenceConfig(environment = myenv, entry_script = "D. score.py")
aci_config = AciWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 4)


### Load dataset
hd_ds = Dataset.get_by_name(ws, "heart_disease_data")


### Construct pipeline
data_store = ws.get_default_datastore()
prepped_data = OutputFileDatasetConfig("prepped")

# Prepare step
step1 = PythonScriptStep(name = "prepare_data",
                         source_directory = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/ml-ci/code/Users/judyxdu/Heart-Disease/Code",
                         script_name = "B. prep.py",
                         compute_target = compute_target,
                         runconfig = runconfig,
                         arguments = ["--hs_ds", hd_ds.as_named_input("heart_disease_data"), "--out_folder", prepped_data])

# Train step
step2 = PythonScriptStep(name = "train_data",
                         source_directory = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/ml-ci/code/Users/judyxdu/Heart-Disease/Code",
                         script_name = "C. train.py",
                         compute_target = compute_target,
                         runconfig = runconfig,
                         arguments = ["--prepped_data", prepped_data.as_input()])

# Combine steps
train_pipeline = Pipeline(workspace = ws, steps = [step1, step2])

# Run pipeline
pipeline_run = experiment.submit(train_pipeline, show_output = True)
pipeline_run.wait_for_completion()


### Deploy
os.chdir("/mnt/batch/tasks/shared/LS_root/mounts/clusters/ml-ci/code/Users/judyxdu/Heart-Disease/Code")
model = Model(ws, "cb")
aci_service = Model.deploy(ws, name = "heart-disease-scoring", models = [model], inference_config = inf_config, deployment_config = aci_config)
aci_service.wait_for_deployment(True)
print(aci_service.state)


