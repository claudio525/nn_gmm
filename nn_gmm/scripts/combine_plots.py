import multiprocessing as mp
import argparse
from pathlib import Path
from PIL import Image


def process_img(img_ffp_1: Path, img_ffp_2: Path, output_ffp: Path):
    assert img_ffp_1.name == img_ffp_2.name

    cur_img_1 = Image.open(img_ffp_1)
    cur_img_2 = Image.open(img_ffp_2)

    width = cur_img_1.size[0] + cur_img_2.size[0]

    new_im = Image.new("RGB", (width, cur_img_1.size[1]))
    new_im.paste(cur_img_1, (0, 0))
    new_im.paste(cur_img_2, (cur_img_1.size[0], 0))

    new_im.save(output_ffp)


def main(dir_1: Path, dir_2: Path, output_dir: Path, n_procs: int = 8):
    plot_ffps_1, plot_ffps_2 = list(dir_1.glob("*.png")), list(dir_2.glob("*.png"))

    # Get all plots with the same names
    plot_names_1 = set([cur_ffp.name for cur_ffp in plot_ffps_1])
    plot_names_2 = set([cur_ffp.name for cur_ffp in plot_ffps_2])

    common_names = plot_names_1.intersection(plot_names_2)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with mp.Pool(processes=n_procs) as pool:
        pool.starmap(
            process_img,
            [
                (dir_1 / cur_name, dir_2 / cur_name, output_dir / cur_name)
                for cur_name in common_names
            ],
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dir_1", help="Base directory 1", type=str)
    parser.add_argument("dir_2", help="Base directory 2", type=str)
    parser.add_argument("output_dir", help="Output directory", type=str)

    args = parser.parse_args()

    main(Path(args.dir_1), Path(args.dir_2), Path(args.output_dir))
