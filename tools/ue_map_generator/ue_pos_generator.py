import argparse

import numpy as np
import pandas as pd


def generate_ue(input_file: str, cluster_file: str, output_file: str):
    data = pd.read_csv(f"ue_count_map/{input_file}.csv", header=None).values
    cluster_data = (
        np.zeros_like(data)
        if cluster_file is None
        else pd.read_csv(f"clusters_map/{cluster_file}.csv", header=None).values
    )
    assert data.shape == cluster_data.shape
    ue = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data[i, j]):
                ue.append([cluster_data[i, j], i, j])
    pd.DataFrame(
        [[i] + pos for i, pos in enumerate(ue)],
        columns=["ue_id", "cluster_id", "x", "y"],
    ).to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("-c", "--cluster-file", type=str, default=None)
    parser.add_argument("-o", "--output-file", type=str, default="generated_ue.csv")
    args = parser.parse_args()
    generate_ue(args.input_file, args.cluster_file, args.output_file)


if __name__ == "__main__":
    main()
