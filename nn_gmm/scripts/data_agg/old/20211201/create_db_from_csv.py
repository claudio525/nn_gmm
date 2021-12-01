from argparse import ArgumentParser
from pathlib import Path

import ml_tools
import nn_gmm


def main(output_ffp: Path, csv_data_dir: Path, filter_file_ffp: Path = None):
    csv_ffps = list(csv_data_dir.glob("*.csv"))
    if filter_file_ffp is not None:
        csv_ffps = [
            cur_ffp
            for cur_ffp in csv_ffps
            if cur_ffp.stem in ml_tools.utils.load_txt(filter_file_ffp)
        ]

    nn_gmm.TableDB.from_csv(output_ffp, csv_ffps)


if __name__ == "__main__":
    parser = ArgumentParser(description="Creates a TableDB from csv files")
    parser.add_argument("output_ffp", type=Path, help="Output file path")
    parser.add_argument(
        "csv_data_dir", type=Path, help="Data directory that contains the csv files"
    )
    parser.add_argument(
        "--filter_file_ffp",
        type=Path,
        help="Text file that contains the csv files to use (without extension), otherwise all are used",
        default=None,
    )

    args = parser.parse_args()

    main(args.output_ffp, args.csv_data_dir, filter_file_ffp=args.filter_file_ffp)
