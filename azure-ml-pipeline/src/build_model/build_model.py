import os
import pickle
from argparse import ArgumentParser

import pandas as pd
import xgboost as xgb
from azureml.core.run import Run
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

training_split = 0.25


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    run_context = Run.get_context()
    df = run_context.input_datasets["ds_processed_data"].to_pandas_dataframe()

    # Numerical encoding of categorical feature:
    # lbl = preprocessing.LabelEncoder()
    # df["type"] = lbl.fit_transform(df["type"].astype(str))

    # Separate feature variables and target variable
    X = df.drop(["quality", "goodquality"], axis=1)
    y = df["goodquality"]

    # Normalize feature variables
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=training_split, random_state=0
    )

    model = xgb.XGBClassifier(random_state=1)
    model.fit(X_train, y_train)
    print ("Training complete")

    model_file_path = os.path.join(args.output_path, "model.dat")

    print("Saving model to " + model_file_path)

    if args.output_path:
        print("Creating folder: " + args.output_path)
        os.makedirs(args.output_path, exist_ok=True)

    with open(model_file_path, "wb") as w:
        print("Dumping model file via pickle")
        pickle.dump(model, w)


if __name__ == "__main__":
    main()
