import re
import uuid
from pathlib import Path
from typing import Any, Optional, Union

from IPython import get_ipython
from suite2p.io.server import ssh_connect

from .io import test_extract_result_present


def extract_job(
    data_info: dict[str, Any],
    settings: dict[str, Any],
    data_path: Union[str, Path],
    force_recalc: bool = False,
    bsubargs: str = '',
    extract_job_path: str = '~/.multiday-suite2p/extract_session_job.sh'
) -> Optional[dict[str, Any]]:
    """
    Submit an extraction job to an LSF cluster using bsub, if results are not already present or recalculation is forced.

    Args:
        data_info (dict[str, Any]): Data info dictionary with data and output folder paths.
        settings (dict[str, Any]): Settings dictionary with server connection info.
        data_path (Union[str, Path]): Path to the session (e.g., 'YYYY_MM_DD/session_id').
        force_recalc (bool, optional): If True, force recalculation even if results exist. Defaults to False.
        bsubargs (str, optional): Additional arguments for bsub. Defaults to ''.
        extract_job_path (str, optional): Path to the extraction job script. Defaults to '~/.multiday-suite2p/extract_session_job.sh'.

    Returns:
        Optional[dict[str, Any]]: dictionary with 'data_path' and 'job_id' if job is submitted, else None.

    Raises:
        NameError: If job submission fails or job id cannot be found.
    """
    result_folder = Path(data_info['data']['local_processed_root']) / data_info['data']['output_folder'] / 'sessions' / data_path
    if (not test_extract_result_present(result_folder)) or force_recalc:
        multiday_linux = (Path(data_info['data']['server_processed_root']) / data_info['data']['output_folder']).as_posix()
        data_folder = data_info['data']['server_processed_root']
        bin_folder = data_info['data']['server_bin_root']
        data_path = Path(data_path)
        server = settings['server']
        date = data_path.parts[-2]
        session = data_path.parts[-1]
        job_id_str = f'{date}-{session}-{uuid.uuid4().hex[:3].upper()}'
        n_cores = server['n_cores']
        ssh = ssh_connect(server['host'], server['username'], server['password'], verbose=False)
        # Create the logs directory on server if it doesn't exist
        ssh.exec_command(f'mkdir -p logs')
        run_command = (
            f'bsub -n {n_cores} -J {job_id_str} {bsubargs} -o logs/extract-{job_id_str}.txt "source {extract_job_path}'
            f' \'{multiday_linux}\''
            f' \'{data_folder}\''
            f' \'{bin_folder}\''
            f' \'{data_path.as_posix()}\''
            f' > logs/log-extract-{job_id_str}.txt"'
        )
        stdin, stdout, stderr = ssh.exec_command(run_command)
        stdout_str = stdout.read().decode('utf-8')
        match = re.search(r'Job <(\d+)>', stdout_str)
        if match:
            job_id = int(match.group(1))
            print(f'{data_path} - job: {job_id}')
            return {'data_path': str(data_path), 'job_id': job_id}
        else:
            raise NameError("Could not find job id (was job submit successful?)")
    else:
        print(f'{data_path} - Already present')
    return None

def check_job_status(
    job: dict[str, Any],
    data_info: dict[str, Any],
    settings: dict[str, Any]
) -> None:
    """
    Check the status of a job submitted to an LSF cluster using bjobs.

    Args:
        job (dict[str, Any]): dictionary with 'job_id' and 'data_path'.
        data_info (dict[str, Any]): Data info dictionary.
        settings (dict[str, Any]): Settings dictionary with server connection info.
    """
    server = settings['server']
    ssh = ssh_connect(server['host'], server['username'], server['password'], verbose=False)
    stdin, stdout, stderr = ssh.exec_command(f'bjobs -l {job["job_id"]}')
    status_match = re.search(r"Status <.*>", stdout.read().decode('utf-8'))
    status = status_match.group() if status_match else 'Status <Unknown>'
    result_folder = Path(data_info['data']['local_processed_root']) / data_info['data']['output_folder'] / 'sessions' / job["data_path"]
    file_status = 'present' if test_extract_result_present(result_folder) else 'absent'
    print(f'{job["data_path"]}: {status}, trace files: {file_status}')

def extract_job_slurm(
    data_info: dict[str, Any],
    settings: dict[str, Any],
    data_path: Union[str, Path],
    force_recalc: bool = False,
    sbatchargs: str = '',
    extract_job_path: str = '~/.multiday-suite2p/extract_session_job.sh'
) -> Optional[dict[str, Any]]:
    """
    Submit an extraction job to a SLURM cluster using sbatch, if results are not already present or recalculation is forced.

    Args:
        data_info (dict[str, Any]): Data info dictionary with data and output folder paths.
        settings (dict[str, Any]): Settings dictionary with server connection info.
        data_path (Union[str, Path]): Path to the session (e.g., 'YYYY_MM_DD/session_id').
        force_recalc (bool, optional): If True, force recalculation even if results exist. Defaults to False.
        sbatchargs (str, optional): Additional arguments for sbatch. Defaults to ''.
        extract_job_path (str, optional): Path to the extraction job script. Defaults to '~/.multiday-suite2p/extract_session_job.sh'.

    Returns:
        Optional[dict[str, Any]]: dictionary with 'data_path' and 'job_id' if job is submitted, else None.

    Raises:
        NameError: If job submission fails or job id cannot be found.
    """
    result_folder = Path(data_info['data']['local_processed_root']) / data_info['data']['output_folder'] / 'sessions' / data_path
    if (not test_extract_result_present(result_folder)) or force_recalc:
        multiday_linux = (Path(data_info['data']['server_processed_root']) / data_info['data']['output_folder']).as_posix()
        data_folder = data_info['data']['server_processed_root']
        bin_folder = data_info['data']['server_bin_root']
        data_path = Path(data_path)
        server = settings['server']
        date = data_path.parts[-2]
        session = data_path.parts[-1]
        job_id_str = f'{date}-{session}-{uuid.uuid4().hex[:3].upper()}'
        n_cores = server['n_cores']
        ssh = ssh_connect(server['host'], server['username'], server['password'], verbose=True)
        run_command = (
            f'sbatch -n {n_cores} -J "{job_id_str}" {sbatchargs} -o "logs/extract-{job_id_str}.txt" --wrap="source {extract_job_path}'
            f" '{multiday_linux}'"
            f" '{data_folder}'"
            f" '{bin_folder}'"
            f" '{data_path.as_posix()}'"
            f' > logs/log-extract-{job_id_str}.txt"'
        )
        stdin, stdout, stderr = ssh.exec_command(run_command)
        stdout_str = stdout.read().decode('utf-8')
        match = re.search(r'Submitted batch job (\d+)', stdout_str)
        if match:
            job_id = int(match.group(1))
            print(f'{data_path} - job: {job_id}')
            return {'data_path': str(data_path), 'job_id': job_id}
        else:
            raise NameError("Could not find job id (was job submit successful?)")
    else:
        print(f'{data_path} - Already present')
    return None

def check_job_status_slurm(
    job: dict[str, Any],
    data_info: dict[str, Any],
    settings: dict[str, Any]
) -> None:
    """
    Check the status of a job submitted to a SLURM cluster using scontrol.

    Args:
        job (dict[str, Any]): dictionary with 'job_id' and 'data_path'.
        data_info (dict[str, Any]): Data info dictionary.
        settings (dict[str, Any]): Settings dictionary with server connection info.
    """
    server = settings['server']
    ssh = ssh_connect(server['host'], server['username'], server['password'], verbose=False)
    stdin, stdout, stderr = ssh.exec_command(f'scontrol show job {job["job_id"]}')
    status_output = stdout.read().decode('utf-8')
    status_match = re.search(r"JobState=(\w+)", status_output)
    status = f"Status <{status_match.group(1)}>" if status_match else "Status <Unknown>"
    result_folder = Path(data_info['data']['local_processed_root']) / data_info['data']['output_folder'] / 'sessions' / job["data_path"]
    file_status = 'present' if test_extract_result_present(result_folder) else 'absent'
    print(f'{job["data_path"]}: {status}, trace files: {file_status}')

def extract_job_ipython(
    data_info: dict[str, Any],
    settings: dict[str, Any],
    data_path: Union[str, Path],
    force_recalc: bool = False,
    sbatchargs: str = '',
    extract_job_path: str = '~/.multiday-suite2p/extract_session_job.sh'
) -> Optional[dict[str, Any]]:
    """
    Submit an extraction job using sbatch via IPython system call, if results are not already present or recalculation is forced.

    Args:
        data_info (dict[str, Any]): Data info dictionary with data and output folder paths.
        settings (dict[str, Any]): Settings dictionary with server connection info.
        data_path (Union[str, Path]): Path to the session (e.g., 'YYYY_MM_DD/session_id').
        force_recalc (bool, optional): If True, force recalculation even if results exist. Defaults to False.
        sbatchargs (str, optional): Additional arguments for sbatch. Defaults to ''.
        extract_job_path (str, optional): Path to the extraction job script. Defaults to '~/.multiday-suite2p/extract_session_job.sh'.

    Returns:
        Optional[dict[str, Any]]: dictionary with 'data_path' and 'job_id' if job is submitted, else None.

    Raises:
        NameError: If job submission fails or job id cannot be found.
    """
    result_folder = Path(data_info['data']['local_processed_root']) / data_info['data']['output_folder'] / 'sessions' / data_path
    if (not test_extract_result_present(result_folder)) or force_recalc:
        multiday_linux = (Path(data_info['data']['server_processed_root']) / data_info['data']['output_folder']).as_posix()
        data_folder = data_info['data']['server_processed_root']
        bin_folder = data_info['data']['server_bin_root']
        data_path = Path(data_path)
        date = data_path.parts[-2]
        session = data_path.parts[-1]
        job_id_str = f'{date}-{session}-{uuid.uuid4().hex[:3].upper()}'
        n_cores = settings['server']['n_cores']
        # Create the logs directory on server if it doesn't exist
        get_ipython().getoutput(f'mkdir -p logs')
        run_command = (
            f'sbatch -n {n_cores} -J "{job_id_str}" {sbatchargs} -o logs/extract-{job_id_str}.txt '
            f'--wrap="source {extract_job_path}'
            f" '{multiday_linux}'"
            f" '{data_folder}'"
            f" '{bin_folder}'"
            f" '{data_path.as_posix()}'"
            f' > logs/log-extract-{job_id_str}.txt"'
        )
        print("Running command:", run_command)
        output = get_ipython().getoutput(run_command)
        output_str = "\n".join(output)
        match = re.search(r"Submitted batch job (\d+)", output_str)
        if match:
            submitted_job_id = int(match.group(1))
            print(f'{data_path} - job: {submitted_job_id}')
            return {'data_path': str(data_path), 'job_id': submitted_job_id}
        else:
            raise NameError("Could not find job id (was job submit successful?)")
    else:
        print(f'{data_path} - Already present')
    return None

def check_job_status_ipython(
    job: dict[str, Any],
    data_info: dict[str, Any]
) -> None:
    """
    Check the status of a job submitted via IPython system call using scontrol.

    Args:
        job (dict[str, Any]): dictionary with 'job_id' and 'data_path'.
        data_info (dict[str, Any]): Data info dictionary.
    """
    command = f'scontrol show job {job["job_id"]}'
    print("Checking job status with:", command)
    output = get_ipython().getoutput(command)
    output_str = "\n".join(output)
    status_match = re.search(r"JobState=(\w+)", output_str)
    status = f"Status <{status_match.group(1)}>" if status_match else "Status <Unknown>"
    result_folder = Path(data_info['data']['local_processed_root']) / data_info['data']['output_folder'] / 'sessions' / job["data_path"]
    file_status = 'present' if test_extract_result_present(result_folder) else 'absent'
    print(f'{job["data_path"]}: {status}, trace files: {file_status}')
