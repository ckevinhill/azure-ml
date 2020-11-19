import azureml.core
from azureml.core import Dataset, Datastore, Environment, Experiment, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

from utils.factory import AzureMLFactory

""" Example Pipeline:

    1. Reference data from dataset (Downloaded from: https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
    2. Create classification column "wine quality" (if quality > 7)
    3. Train XGB Classifier to predict wine quality

"""

# Get configured workspace from .azureml/config.json:
ws = Workspace.from_config()

# Set-up execution environment:
run_config = RunConfiguration()

# Get compute resource:
compute_name = "ml-compute-inst"
compute_target = AzureMLFactory.getComputeInstanceResource(ws, compute_name)
run_config.target = compute_target

# Get environment (dependencies):
environment_name = "ml-env"
environment = AzureMLFactory.getEnvironment(
    ws, environment_name, "src/create_classification_target/requirements.txt"
)
run_config.environment = environment

# --------------------------------------------------------------

# Define first pipeline step:
ds_processed_data = PipelineData(
    "ds_processed_data", datastore=ws.get_default_datastore()
)

dataset = Dataset.get_by_name(ws, name="wine-quality")
entry_point = "create_classification_target/create_classification_step.py"
data_processing_step = PythonScriptStep(
    script_name=entry_point,
    source_directory="./src,
    arguments=["--output_path", ds_processed_data],
    inputs=[dataset.as_named_input("ds_input")],
    outputs=[ds_processed_data],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=True,
)

ds_model = PipelineData(
    "ds_model", datastore=ws.get_default_datastore()
)
entry_point = "build_model/build_model.py"
model_build_step = PythonScriptStep(
    script_name=entry_point,
    source_directory="./src,
    arguments=["--output_path", ds_model],
    inputs=[ds_processed_data],
    outputs=[ds_model],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=True,
)
steps = [model_build_step]

pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

pipeline_run = Experiment(ws, "pipeline_s_experiment").submit(pipeline)
pipeline_run.wait_for_completion()
