from pathlib import Path
from shutil import copyfile

import click

from autoalbument.utils.templates import AutoAlbumentTemplate


@click.command()
@click.option(
    "--path", type=click.Path(), required=True, help="Path to a directory where AutoAlbument will place config files."
)
@click.option("--num-classes", type=int, required=True, help="Number of classes in the dataset.")
def main(path, num_classes):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    templates_dir = Path(__file__).parent.parent / "faster_autoaugment" / "templates"
    dataset_file = templates_dir / "dataset.py.tmpl"
    search_config_file = templates_dir / "search.yaml.tmpl"
    dataset_file_destination = path / "dataset.py"
    search_file_destination = path / "search.yaml"

    with search_config_file.open() as f:
        config = AutoAlbumentTemplate(f.read())

    with search_file_destination.open("w") as f:
        f.write(config.substitute(num_classes=num_classes))

    copyfile(dataset_file, dataset_file_destination)

    click.echo()

    click.echo(
        f"Files dataset.py and search.yaml are created in {path}.\n\n"
        f"Next steps:\n"
        f"1. Add the required implementation for dataset methods in "
        + click.style(str(dataset_file_destination), bold=True)
        + "\n"
        + "2. [Optional] Adjust search parameters in "
        + click.style(str(search_file_destination), bold=True)
        + "\n"
        + "3. Run AutoAlbument search with the following command:\n\n"
        + click.style(f"autoalbument-search --config-dir {path}\n", bold=True)
    )


if __name__ == "__main__":
    main()
