from azureml.core import Environment, Workspace
from azureml.core.compute import AmlCompute, ComputeInstance, ComputeTarget
from azureml.pipeline.core import Pipeline


class AzureMLFactory:
    @staticmethod
    def getComputeInstanceResource(
        ws: Workspace, compute_name: str, vm_size: str = "Standard_DS1_v2"
    ):

        if compute_name in ws.compute_targets:
            compute_target = ws.compute_targets[compute_name]
            if compute_target and type(compute_target) is ComputeInstance:
                return compute_target

        # Create new compute resource:
        provisioning_config = ComputeInstance.provisioning_configuration(
            vm_size=vm_size, ssh_public_access=False
        )

        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

        compute_target.wait_for_completion(show_output=True)

        return compute_target

    @staticmethod
    def getComputeClusterResource(
        ws: Workspace,
        compute_name: str,
        vm_size: str = "Standard_DS1_v2",
        min_nodes=0,
        max_nodes=1,
    ):

        if compute_name in ws.compute_targets:
            compute_target = ws.compute_targets[compute_name]
            if compute_target and type(compute_target) is AmlCompute:
                return compute_target

        # Create new compute resource:
        provisioning_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, min_nodes=min_nodes, max_nodes=max_nodes
        )

        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20
        )

        return compute_target

    @staticmethod
    def getEnvironment(
        ws: Workspace, environment_name: str, requirements_file, create_new=False
    ):

        environments = Environment.list(workspace=ws)

        # Retrieve existing environment:
        if not create_new:
            existing_environment = None
            for env in environments:
                if env == environment_name:
                    return environments[environment_name]

        # Create new environment:
        new_environment = Environment.from_pip_requirements(
            environment_name,
            requirements_file,
        )

        new_environment.register(ws)
        return new_environment

    @staticmethod
    def getPipeline(ws: Workspace, pipeline_name):

        pipelines = Pipeline
