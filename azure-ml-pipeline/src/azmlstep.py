import os
from argparse import ArgumentParser

from azureml.core import Dataset, Run, Workspace


class Provider:
    kwargs = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_data_set(self, name):
        raise NotImplementedError("Provider is not implemented")


class RunContextProvider(Provider):
    """Retrieves reference to named DataSet via
    the current run_context"""

    run_context = None

    def __init__(self, **kwargs):
        self.run_context = Run.get_context()
        super().__init__(**kwargs)

    def get_data_set(self, name):
        # TODO: Should this be cached?
        return self.run_context.input_datasets[name]


class LocalFileSystemProvider(Provider):
    """Retrieves reference to named DataSet via
    upload of file on local file system"""

    data_directory = ""
    target_directory = ""
    overwrite = "False"
    workspace = None

    def __init__(
        self,
        workspace,
        overwrite=False,
        data_directory="",
        target_directory="",
        **kwargs
    ):
        self.workspace = workspace
        self.overwrite = overwrite
        self.data_directory = data_directory
        self.target_directory = target_directory

        super().__init__(**kwargs)

    def get_data_set(self, name):
        file_path = os.path.join(self.data_directory, name + ".csv")

        # Upload data to datastore:
        datastore = self.workspace.get_default_datastore()
        datastore.upload_files(
            [file_path], target_path=self.target_directory, overwrite=self.overwrite
        )

        # Return Dataset:
        return Dataset.Tabular.from_delimited_files(
            path=[(datastore, (self.target_directory + "/" + name + ".csv"))]
        )


class Requirement:
    """A Descriptor that represents a named
    data input for an AzureMLStep."""

    name = ""
    data_set_provider = None
    cache = True
    kwargs = {}
    dataset = None

    def __init__(self, provider=RunContextProvider, cache=True, **kwargs):
        self.data_set_provider = provider
        self.kwargs = kwargs
        self.cache = cache

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if self.cache and self.dataset:
            return self.dataset

        self.dataset = self.data_set_provider(**self.kwargs).get_data_set(self.name)

        return self.dataset


class AzureMLStep:

    output_path = ""

    def __init__(self, output_arg="output_path", alt_args=None, **kwargs):
        parser = ArgumentParser()
        parser.add_argument("--" + output_arg, default="", type=str)
        args = parser.parse_args()
        if alt_args:
            args = parser.parse_args(alt_args)

        self.output_path = getattr(args, output_arg)

    def output(self):
        os.makedirs(self.output_path, exist_ok=True)
        return self.output_path
