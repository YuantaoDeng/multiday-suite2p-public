import click
import pickle
import numpy as np
from pathlib import Path

# Example imports from the notebook
from multiday_suite2p.cluster.jobs import extract_job
from multiday_suite2p.io import import_sessions, export_masks_and_images
from multiday_suite2p.process import extract_local
from multiday_suite2p.settings import parse_settings, parse_data_info
from multiday_suite2p.transform import (
    register_sessions,
    transform_cell_masks,
    cluster_cell_masks,
    create_template_masks,
    backward_transform_masks
)


@click.group()
def cli():
    """CLI for running multi-day registration tasks."""
    pass


@cli.command()
@click.option("--data-info", type=click.Path(exists=True), required=True, help="Path to data info file.")
@click.option("--settings-file", type=click.Path(exists=True), required=True, help="Path to settings file.")
@click.option("--outdir", type=click.Path(), default=".", help="Folder where output is saved.")
def import_data(data_info, settings_file, outdir):
    """Imports session data and saves intermediate results."""
    data_info_obj = parse_data_info(data_info)
    settings_obj = parse_settings(settings_file, request_pass=True)
    sessions, images, cells, im_size, label_im = import_sessions(data_info_obj, settings_obj, verbose=True)

    outpath = Path(outdir) / "import.pkl"
    with open(outpath, "wb") as f:
        pickle.dump([sessions, images, cells, im_size, label_im], f)
    click.echo(f"Imported sessions saved to {outpath}")


@cli.command()
@click.option("--import-pkl", type=click.Path(exists=True), required=True, help="Pickle file from import_data step.")
@click.option("--settings-file", type=click.Path(exists=True), required=True, help="Path to settings file.")
def register_data(import_pkl, settings_file):
    """Registers session images, transforms masks, and saves output."""
    with open(import_pkl, "rb") as f:
        sessions, images, cells, im_size, label_im = pickle.load(f)
    settings_obj = parse_settings(settings_file, request_pass=True)

    deforms, trans_images = register_sessions(images, settings_obj['registration'])
    trans_masks, trans_label_im = transform_cell_masks(deforms, cells)

    reg_pkl = Path(import_pkl).parent / "register.pkl"
    with open(reg_pkl, "wb") as f:
        pickle.dump([deforms, trans_images, trans_masks, trans_label_im], f)
    click.echo(f"Registration output saved to {reg_pkl}")


@cli.command()
@click.option("--register-pkl", type=click.Path(exists=True), required=True, help="Pickle file from register_data step.")
@click.option("--settings-file", type=click.Path(exists=True), required=True, help="Path to settings file.")
def cluster_masks_cmd(register_pkl, settings_file):
    """Clusters registered cell masks and creates a template mask."""
    with open(register_pkl, "rb") as f:
        deforms, trans_images, trans_masks, trans_label_im = pickle.load(f)
    settings_obj = parse_settings(settings_file, request_pass=True)

    # Cluster
    im_size = trans_images[0]['raw'].shape if trans_images else (512, 512)
    matched_cells, matched_im = cluster_cell_masks(trans_masks, im_size, settings_obj['clustering'])
    template_masks, template_im = create_template_masks(matched_cells, im_size, settings_obj['clustering'])

    match_pkl = Path(register_pkl).parent / "match.pkl"
    with open(match_pkl, "wb") as f:
        pickle.dump([matched_cells, matched_im, template_masks, template_im], f)
    click.echo(f"Clustering output saved to {match_pkl}")


@cli.command()
@click.option("--match-pkl", type=click.Path(exists=True), required=True, help="Pickle file from cluster_masks_cmd step.")
@click.option("--register-pkl", type=click.Path(exists=True), required=True, help="Pickle file from register_data step.")
def backward_transform(match_pkl, register_pkl):
    """Backward-transforms the filtered template masks to each session's original space."""
    with open(match_pkl, "rb") as fm:
        matched_cells, matched_im, template_masks, template_im = pickle.load(fm)
    with open(register_pkl, "rb") as fr:
        deforms, trans_images, trans_masks, trans_label_im = pickle.load(fr)

    deform_masks, deform_label_ims, deform_lam_ims = backward_transform_masks(template_masks, deforms)

    filter_pkl = Path(match_pkl).parent / "filter.pkl"
    with open(filter_pkl, "wb") as f:
        pickle.dump([deform_masks, deform_label_ims, deform_lam_ims, template_masks], f)
    click.echo(f"Backward transform output saved to {filter_pkl}")


@cli.command()
@click.option("--filter-pkl", type=click.Path(exists=True), required=True, help="Pickle file from backward_transform.")
@click.option("--import-pkl", type=click.Path(exists=True), required=True, help="Pickle file from import_data step.")
@click.option("--register-pkl", type=click.Path(exists=True), required=True, help="Pickle file from register_data step.")
@click.option("--data-info", type=click.Path(exists=True), required=True, help="Data info file for export.")
@click.option("--settings-file", type=click.Path(exists=True), required=True, help="Settings file for export.")
def export_results(filter_pkl, import_pkl, register_pkl, data_info, settings_file):
    """Exports new cell masks and images."""
    data_info_obj = parse_data_info(data_info)
    settings_obj = parse_settings(settings_file, request_pass=True)

    with open(import_pkl, "rb") as fi:
        sessions, images, cells, im_size, label_im = pickle.load(fi)
    with open(register_pkl, "rb") as fr:
        deforms, trans_images, trans_masks, trans_label_im = pickle.load(fr)
    with open(filter_pkl, "rb") as ff:
        deform_masks, deform_label_ims, deform_lam_ims, template_masks = pickle.load(ff)

    export_masks_and_images(deform_masks, template_masks, trans_images, images, sessions, data_info_obj, settings_obj)
    click.echo("Export completed.")

@cli.command()
@click.option("--data-info", type=click.Path(exists=True), required=True, help="Path to data info file.")
@click.option("--settings-file", type=click.Path(exists=True), required=True, help="Path to settings file.")
@click.option("--force-recalc", is_flag=True, default=False, help="Force trace extraction even if result files exist.")
@click.option("--use-server", is_flag=True, default=False, help="If True, submit jobs to the server using extract_job, otherwise run locally using extract_local.")
def extract_traces(data_info: str, settings_file: str, force_recalc: bool, use_server: bool) -> None:
    """
    Extract fluorescence traces for all sessions, either locally or by submitting jobs to a cluster/server.

    Args:
        data_info (str): Path to a .json or .yaml file with data info (points to session directories).
        settings_file (str): Path to a .json or .yaml file with settings.
        force_recalc (bool): If True, recalculate even if existing trace files are found.
        use_server (bool): If True, calls extract_job to submit trace extraction to a cluster/server;
                           otherwise uses extract_local to run extraction in this environment.
    """
    data_info_obj = parse_data_info(data_info)
    settings_obj = parse_settings(settings_file, request_pass=True)

    # We assume that 'info.npy' has a 'data_paths' key listing session directories
    info_file = Path(data_info_obj["data"]["local_processed_root"]) / data_info_obj["data"]["output_folder"] / "info.npy"
    info_dict = np.load(info_file, allow_pickle=True).item()

    if use_server:
        # Submit to server
        click.echo("Submitting extraction jobs to the server...")
        for data_path in info_dict["data_paths"]:
            outcome = extract_job(data_info_obj, settings_obj, data_path, force_recalc=force_recalc)
            click.echo(f"extract_job -> {outcome}")
    else:
        # Run locally
        click.echo("Running extraction locally...")
        for data_path in info_dict["data_paths"]:
            outcome = extract_local(data_info_obj, data_path, force_recalc=force_recalc)
            click.echo(f"extract_local -> {outcome}")

    click.echo("Extraction process completed.")


if __name__ == "__main__":
    cli()
