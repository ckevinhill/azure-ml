import os
from argparse import ArgumentParser

from azureml.core.run import Run


def create_classification_target(df):
    df["goodquality"] = [1 if x >= 7 else 0 for x in df["quality"]]
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    run_context = Run.get_context()
    df = run_context.input_datasets["ds_input"].to_pandas_dataframe()

    df = create_classification_target(df)

    if not args.output_path:
        return

    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, "output.parquet")
    df.to_parquet(output_path)


if __name__ == "__main__":
    main()
