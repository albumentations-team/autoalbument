from pathlib import Path
from shutil import copyfile

import click

from autoalbument.cli.lib.migrations import migrate_v1_to_v2
from autoalbument.cli.lib.yaml import yaml
from autoalbument.utils.click import should_write_file


@click.command()
@click.option(
    "--config-dir",
    type=click.Path(),
    required=True,
    help="Path to a directory with search.yaml",
)
def main(config_dir):
    config_dir = Path(config_dir)
    config_path = config_dir / "search.yaml"
    config_backup_path = config_dir / "search.yaml.backup"

    config = yaml.load(config_path)
    version = config.get("_version", 1)
    if version == 2:
        click.echo("search.yaml is already uses the latest config format")
        return

    click.echo("Backing up the original search.yaml file to search.yaml.backup")

    if should_write_file(config_backup_path):
        copyfile(config_path, config_backup_path)

    config = migrate_v1_to_v2(config)
    yaml.dump(config, config_path)

    click.echo("search.yaml is successfully updated to the latest config format")


if __name__ == "__main__":
    main()
