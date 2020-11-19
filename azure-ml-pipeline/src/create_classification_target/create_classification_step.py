import os
from argparse import ArgumentParser

from azmlstep import AzureMLStep, Requirement, RunContextProvider
from azureml.core.run import Run


class CreateClassificationStep(AzureMLStep):

    # The expected input value:
    ds_input = Requirement()
    output_filename = "output.parquet"

    def run(self):
        df = self.ds_input.to_pandas_dataframe()
        df["goodquality"] = [1 if x >= 7 else 0 for x in df["quality"]]
        df.to_parquet(os.path.join(self.output(), self.output_filename))


def main():
    step = CreateClassificationStep()
    step.run()


if __name__ == "__main__":
    main()
