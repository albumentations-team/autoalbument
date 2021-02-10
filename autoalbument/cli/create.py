from pathlib import Path
from shutil import copyfile

import click

from autoalbument.cli.lib.config_builder import SearchConfigBuilder
from autoalbument.utils.click import should_write_file


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
@click.option(
    "--generate-full-config",
    is_flag=True,
    help="Create a config file that contains all available configuration parameters (by default, the config file will"
    "contain only the most important parameters).",
)
def main(config_dir, task, num_classes, generate_full_config):
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    resources_dir = Path(__file__).parent / "resources"
    dataset_file = resources_dir / f"{task}_dataset.py.tmpl"
    dataset_file_destination = config_dir / "dataset.py"
    search_file_destination = config_dir / "search.yaml"

    base_config_path = Path(__file__).parent / "conf" / "config.yaml"
    short_config_keys_path = resources_dir / "short_config_keys.yaml"
    search_config_builder = SearchConfigBuilder(
        base_config_path,
        short_config_keys_path=short_config_keys_path,
        task=task,
        num_classes=num_classes,
        generate_full_config=generate_full_config,
    )

    if should_write_file(search_file_destination):
        search_config_builder.write_config(search_file_destination)

    if should_write_file(dataset_file_destination):
        copyfile(dataset_file, dataset_file_destination)

    click.echo(
        f"\nFiles dataset.py and search.yaml are placed in {config_dir}.\n\n"
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
