import os
import sys
import unittest
from unittest import TestCase

import pandas as pd
from azureml.core import Dataset

from src.azmlstep import (AzureMLStep, LocalFileSystemProvider, Provider,
                          Requirement)


class MockStep(AzureMLStep):

    # Input for the AzureMLStep
    test_data = Requirement(
        LocalFileSystemProvider,
        data_directory="data",
        target_directory="data",
        overwrite=True,
    )

    def run(self, output_filename="test_output.csv"):
        df = self.test_data.to_pandas_dataframe()
        df["colD"] = df["colC"] - 1

        df.to_csv(os.path.join(self.output(), output_filename))


def generateTestDataFrame():
    data = {
        "colA": [1, 2, 3, 4, 5],
        "colB": ["A", "B", "C", "D", "E"],
        "colC": [6, 7, 8, 9, 10],
    }

    return pd.DataFrame(data=data)


class AzureMLStepTest(TestCase):

    mockStep = None
    test_dir = "data"
    # File name root should match MockStep attribute:
    test_file = "test_data.csv"

    output_dir = "data/output"
    output_filename = "test_output.csv"

    def setUp(self):

        # Create Step after adjustment to args:
        if not self.mockStep:
            alt_args = ["--output_path", self.output_dir]
            self.mockStep = MockStep(alt_args=alt_args)

        # Create a serialized version of testDataFrame:
        test_file_path = os.path.join(self.test_dir, self.test_file)
        if not os.path.exists(test_file_path):
            df = generateTestDataFrame()
            df.to_csv(os.path.join(self.test_dir, self.test_file), index=False)

    def test_requirement(self):

        test_df = generateTestDataFrame()
        dataset = self.mockStep.test_data
        pd.testing.assert_frame_equal(test_df, dataset.to_pandas_dataframe())

    def test_step(self):

        self.mockStep.run(output_filename=self.output_filename)

        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, self.output_filename))
        )


if __name__ == "__main__":
    unittest.main()
