from pathlib import Path
from shutil import copyfile

import click

from autoalbument.utils.templates import AutoAlbumentTemplate


@click.command()
@click.option(
    "--config-dir",
    type=click.Path(),
    required=True,
    help="Path to a directory where AutoAlbument will place config files.",
)
@click.option(
    "--task",
    type=click.Choice(["classification", "semantic_segmentation"]),
    required=True,
    help="Deep learning task (either classification or semantic segmentation)",
)
@click.option("--num-classes", type=int, required=True, help="Number of classes in the dataset.")
def main(config_dir, task, num_classes):
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = Path(__file__).parent.parent / "faster_autoaugment" / "templates"
    dataset_file = templates_dir / task / "dataset.py.tmpl"
    search_config_file = templates_dir / task / "search.yaml.tmpl"
    dataset_file_destination = config_dir / "dataset.py"
    search_file_destination = config_dir / "search.yaml"

    with search_config_file.open() as f:
        config = AutoAlbumentTemplate(f.read())

    with search_file_destination.open("w") as f:
        f.write(config.substitute(num_classes=num_classes))

    copyfile(dataset_file, dataset_file_destination)

    click.echo()

    click.echo(
        f"Files dataset.py and search.yaml are created in {config_dir}.\n\n"
        f"Next steps:\n"
        f"1. Add the required implementation for dataset methods in "
        + click.style(str(dataset_file_destination), bold=True)
        + "\n"
        + "2. [Optional] Adjust search parameters in "
        + click.style(str(search_file_destination), bold=True)
        + "\n"
        + "3. Run AutoAlbument search with the following command:\n\n"
        + click.style(f"autoalbument-search --config-dir {config_dir}\n", bold=True)
    )


if __name__ == "__main__":
    main()
