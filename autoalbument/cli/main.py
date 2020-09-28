import argparse
import sys
from pathlib import Path
from shutil import copyfile


def create(args):
    destination_path = Path(args.path[0])
    destination_path.mkdir(parents=True, exist_ok=True)
    templates_dir = Path(__file__).parent.parent / "faster_autoaugment" / "templates"
    dataset_file = templates_dir / "dataset.py"
    search_config_file = templates_dir / "search.yaml"
    dataset_file_destination = destination_path / "dataset.py"
    search_file_destination = destination_path / "search.yaml"

    copyfile(dataset_file, dataset_file_destination)
    copyfile(search_config_file, search_file_destination)

    print(
        f"\nFiles dataset.py and search.yaml are created in {destination_path}.\n\n"
        f"Next steps:\n"
        f"1. Add the required implementation for dataset methods in {dataset_file_destination}\n"
        f"2. Update search parameters in {search_file_destination}\n"
        f"3. Run AutoAlbument search with the following command:\n\n"
        f"autoalbument-search --config-dir {destination_path}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    parser_create = subparsers.add_parser(
        "create",
        help="create AutoAlbument experiment files",
    )
    parser_create.set_defaults(func=create)
    parser_create.add_argument(
        "path",
        nargs=1,
    )
    args = parser.parse_args()
    if not args.command:
        parser.parse_args(["--help"])
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
