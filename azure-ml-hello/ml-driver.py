from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace

ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name="hello-world-experiment")

# Setup for a local environment execution:
myenv = Environment("user-managed-env")
myenv.python.user_managed_dependencies = True

# Execute the experiment:
config = ScriptRunConfig(
    source_directory="./src",
    script="hello.py",
    compute_target="local",
    environment=myenv,
)

# Store experiment results:
run = experiment.submit(config)
aml_url = run.get_portal_url()

print("You can access experiment results at:")
print(aml_url)
