import click


def should_write_file(filepath):
    if filepath.exists():
        return click.confirm(f"File {filepath} already exists. Do you want to overwrite it?")
    return True
