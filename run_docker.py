import logging
import os
import threading
from typing import Union

import colors
import docker
import psutil
from docker.models.containers import Container

DOCKER_TAG = "ann-bench"
logging.basicConfig(level=logging.INFO)


def _handle_container_return_value(
    return_value: Union[dict[str, Union[int, str]], int],
    container: Container,
    logger: logging.Logger,
) -> None:
    """Handles the return value of a Docker container and outputs error and stdout messages.

    Args:
        return_value (Union[Dict[str, Union[int, str]], int]): The return value of the container.
        container (docker.models.containers.Container): The Docker container.
        logger (logging.Logger): The logger instance.
    """

    base_msg = f"Child process for container {container.short_id} "
    msg = base_msg + "returned exit code {}"

    if isinstance(
        return_value, dict
    ):  # The return value from container.wait changes from int to dict in docker 3.0.0
        error_msg = return_value.get("Error", "")
        exit_code = return_value["StatusCode"]
        msg = msg.format(f"{exit_code} with message {error_msg}")
    else:
        exit_code = return_value
        msg = msg.format(exit_code)

    if exit_code not in [0, None]:
        logger.error(colors.color(container.logs().decode(), fg="red"))
        logger.error(msg)
    else:
        logger.info(msg)


def run_docker(
    cmd: list[str], cpu_limit: str, mem_limit: int | None = None, timeout: int = 3600
) -> None:
    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available

    container = client.containers.run(
        DOCKER_TAG,
        cmd,
        volumes={
            os.path.dirname(os.path.abspath(__file__)): {
                "bind": "/home/app",
                "mode": "rw",
            },
        },
        cpuset_cpus=str(cpu_limit),
        mem_limit=mem_limit,
        detach=True,
    )
    assert isinstance(container, Container)
    logger = logging.getLogger(f"annb.{container.short_id}")

    logger.info(
        "Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s",
        container.short_id,
        cpu_limit,
        mem_limit,
        timeout,
        cmd,
    )

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        return_value = container.wait(timeout=timeout)
        _handle_container_return_value(return_value, container, logger)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Container.wait for container %s failed with exception", container.short_id
        )
        logger.error(str(e))
    finally:
        logger.info("Removing container")
        container.remove(force=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_hier",
            "--nlist_space",
            "[32]",
            "--gamma1_space",
            "[2]",
            "--gamma2_space",
            "[128]",
            "--max_sl_size_space",
            "[256]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit="0",
        mem_limit=10_000_000_000,
    )
