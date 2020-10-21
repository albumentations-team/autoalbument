import os


def get_hydra_config_dir():
    from hydra.core.global_hydra import GlobalHydra

    load_history = GlobalHydra.instance().config_loader().get_load_history()
    cmd_load_traces = [trace for trace in load_history if trace.provider == "command-line"]
    if not cmd_load_traces:
        cmd_load_traces = [trace for trace in load_history if trace.provider == "main"]
    if not cmd_load_traces:
        raise RuntimeError("Couldn't deduce config_dir from load_history.")
    return cmd_load_traces[-1].path.replace("file://", "")


def get_dataset_filepath(dataset_file):
    if os.path.isabs(dataset_file):
        return dataset_file
    base_path = get_hydra_config_dir()
    return os.path.join(base_path, dataset_file)
