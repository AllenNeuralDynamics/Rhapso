"""
Worker script to run multiscale conversion on a Zarr dataset.
Manages Ray cluster lifecycle similarly to affine_fusion_worker.
"""

import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import dask.array as da  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from Rhapso.fusion.multiscale.aind_z1_radial_correction.array_to_zarr import convert_array_to_zarr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for cleanup
ray_config_path: Optional[str] = None
should_cleanup = False

# Default parameters
DEFAULT_INPUT_ZARR_PATH = "s3://martin-test-bucket/fusion_output/HCR_802704_output_1/channel_488.zarr"
DEFAULT_OUTPUT_ZARR_PATH = "s3://martin-test-bucket/multiscale_output/HCR_802704_output_1/channel_488_multiscale.zarr"
DEFAULT_CHUNK_SIZE = [128, 128, 128]
DEFAULT_VOXEL_SIZE = [1.0, 1.0, 1.0]
DEFAULT_N_LEVELS = 6
DEFAULT_SCALE_FACTOR = [2, 2, 2]
DEFAULT_TARGET_BLOCK_SIZE_MB = 512
DEFAULT_RAY_NUM_CPUS = 12
DEFAULT_RAY_CONFIG_PATH = "Rhapso/pipelines/ray/aws/config/dev/multiscale_cluster_martin.yml"


def cleanup_cluster(yml_filename: str, cwd: Path) -> None:
    """Clean up the Ray cluster and handle any errors gracefully."""
    global should_cleanup
    if should_cleanup:
        return

    print("\n=== Cleaning up cluster ===")
    print("$", " ".join(["ray", "down", yml_filename, "-y"]))
    try:
        subprocess.run(["ray", "down", yml_filename, "-y"], cwd=cwd, capture_output=False, text=True)
        print("✅ Cluster cleanup completed")
    except Exception as cleanup_error:
        print(f"⚠️  Cluster cleanup failed: {cleanup_error}")
        try:
            print("Trying alternative cleanup...")
            subprocess.run(["ray", "down", yml_filename], cwd=cwd, capture_output=False, text=True)
        except Exception:
            print("Alternative cleanup also failed - cluster may need manual cleanup")

    should_cleanup = True


def cleanup_existing_cluster(yml_filename: str, cwd: Path) -> None:
    """Clean up any existing cluster before starting a new one."""
    print("\n=== Clean up any existing cluster ===")
    print("$", " ".join(["ray", "down", yml_filename, "-y"]))
    try:
        subprocess.run(["ray", "down", yml_filename, "-y"], cwd=cwd, capture_output=False, text=True)
        print("✅ Cleanup completed (or no existing cluster)")
    except Exception:
        print("ℹ️  No existing cluster to clean up")


def start_cluster(yml_filename: str, cwd: Path) -> None:
    """Start the Ray cluster defined in the provided YAML file."""
    print("\n=== Start cluster ===")
    print("$", " ".join(["ray", "up", yml_filename, "-y"]))
    try:
        result = subprocess.run(["ray", "up", yml_filename, "-y"], check=True, cwd=cwd, capture_output=False, text=True)
        print("✅ Cluster started successfully")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as error:
        print(f"❌ Cluster startup failed with return code {error.returncode}")
        print("STDOUT:", error.stdout)
        print("STDERR:", error.stderr)
        cleanup_cluster(yml_filename, cwd)
        raise


def signal_handler(sig, frame) -> None:  # type: ignore[override]
    """Handle Ctrl+C gracefully by ensuring the cluster is torn down."""
    global ray_config_path
    print("\n\n⚠️  Interrupt received (Ctrl+C). Cleaning up...")
    if ray_config_path:
        ray_config_dir = Path(ray_config_path).parent
        yml_filename = Path(ray_config_path).name
        cleanup_cluster(yml_filename, ray_config_dir)
    print("✅ Cleanup completed. Exiting.")
    sys.exit(0)


def connect_to_remote_ray_cluster() -> None:
    """Ensure we are connected to an existing Ray cluster when running remotely."""
    try:
        import ray  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - sanity logging
        logger.error("Ray is required for remote execution but is not installed: %s", exc)
        raise

    if ray.is_initialized():
        return

    logger.info("Connecting to existing Ray cluster (address='auto')")
    ray.init(address="auto", ignore_reinit_error=True)
    cpu_count = ray.cluster_resources().get("CPU", 0)  # type: ignore[attr-defined]
    logger.info("Ray connected with %s CPUs available", cpu_count)


def perform_multiscale_conversion(
    *,
    input_zarr_path: str,
    output_zarr_path: str,
    use_ray: bool,
    ray_num_cpus: int,
    chunk_size,
    voxel_size,
    n_lvls: int,
    scale_factor,
    target_block_size_mb: int,
    remote_cluster: bool = False,
) -> None:
    """Core multiscale conversion workflow shared between local and remote runs."""

    start_time = time.time()

    print(f'{input_zarr_path=}')

    dont_copy_fullscale = input_zarr_path == output_zarr_path

    if use_ray and remote_cluster:
        connect_to_remote_ray_cluster()

    logger.info("Starting multiscale conversion")
    logger.info("Input: %s", input_zarr_path)
    logger.info("Output: %s", output_zarr_path)
    if dont_copy_fullscale:
        logger.info("dont_copy_fullscale=True: Will skip writing level 0 and only write levels 1-%s", n_lvls - 1)

    try:
        logger.info("Attempting to load from %s/0...", input_zarr_path)
        sys.stdout.flush()
        dataset = da.from_zarr(f"{input_zarr_path}/0")
        logger.info("Successfully loaded data from %s/0", input_zarr_path)
    except Exception as err:
        logger.warning("Could not load from scale 0: %s", err)
        try:
            logger.info("Attempting to load from %s...", input_zarr_path)
            sys.stdout.flush()
            dataset = da.from_zarr(input_zarr_path)
            logger.info("Successfully loaded data from %s", input_zarr_path)
        except Exception as err_root:
            logger.error("Failed to load data: %s", err_root)
            raise ValueError(
                f"Could not load data from {input_zarr_path} or {input_zarr_path}/0. Error: {err_root}"
            ) from err_root

    logger.info("Dataset shape: %s", dataset.shape)
    logger.info("Dataset dtype: %s", dataset.dtype)
    logger.info("Dataset chunks: %s", dataset.chunks)

    dtype_bytes = np.dtype(dataset.dtype).itemsize
    total_size_gb = np.prod(dataset.shape) * dtype_bytes / (1024**3)
    logger.info("Dataset size: %.2f GB", total_size_gb)

    logger.info("Using Dask array for lazy/chunked processing (not loading into memory)")
    array = dataset

    compressor_kwargs = {
        "cname": "zstd",
        "clevel": 3,
        "shuffle": 2,
    }

    logger.info("=" * 60)
    logger.info("Starting multiscale conversion with parameters:")
    logger.info("  Output path: %s", output_zarr_path)
    logger.info("  Chunk size: %s", chunk_size)
    logger.info("  Voxel size: %s", voxel_size)
    logger.info("  Number of levels: %s", n_lvls)
    logger.info("  Scale factor: %s", scale_factor)
    logger.info("  Target block size: %s MB", target_block_size_mb)
    logger.info("    → Each block will be ~%s MB in memory", target_block_size_mb)
    logger.info("    → Larger blocks = fewer blocks but slower per block")
    logger.info("    → Smaller blocks = more blocks but faster per block")
    logger.info("    → Estimated blocks for level 0: ~%s", int(total_size_gb * 1024 / target_block_size_mb))
    if use_ray:
        logger.info("  Parallel processing: ENABLED (Ray with %s CPUs)", ray_num_cpus)
        logger.info("    → Max parallel memory usage: ~%s MB", ray_num_cpus * target_block_size_mb)
    else:
        logger.info("  Parallel processing: DISABLED (Sequential)")
    logger.info("=" * 60)
    sys.stdout.flush()

    try:
        convert_array_to_zarr(
            array=array,
            chunk_size=chunk_size,
            output_path=output_zarr_path,
            voxel_size=voxel_size,
            n_lvls=n_lvls,
            scale_factor=scale_factor,
            compressor_kwargs=compressor_kwargs,
            target_block_size_mb=target_block_size_mb,
            use_ray=use_ray,
            ray_num_cpus=ray_num_cpus,
            dont_copy_fullscale=dont_copy_fullscale,
        )
    except MemoryError as err_m:
        elapsed_seconds = time.time() - start_time
        logger.error("=" * 60)
        logger.error("MEMORY ERROR: Out of memory!")
        logger.error("This typically happens when target_block_size_mb is too large.")
        logger.error("Try reducing target_block_size_mb (currently %s MB).", target_block_size_mb)
        logger.error("Error details: %s", err_m)
        logger.error("Failed after: %.1f seconds (%.2f minutes)", elapsed_seconds, elapsed_seconds / 60)
        logger.error("=" * 60)
        sys.stdout.flush()
        raise
    except TimeoutError as err_t:
        elapsed_seconds = time.time() - start_time
        logger.error("=" * 60)
        logger.error("TIMEOUT ERROR: Operation timed out!")
        logger.error("This can happen with very large blocks or slow S3 connections.")
        logger.error("Current target_block_size_mb: %s MB", target_block_size_mb)
        logger.error("Error details: %s", err_t)
        logger.error("Failed after: %.1f seconds (%.2f minutes)", elapsed_seconds, elapsed_seconds / 60)
        logger.error("=" * 60)
        sys.stdout.flush()
        raise
    except Exception as err_generic:
        elapsed_seconds = time.time() - start_time
        logger.error("=" * 60)
        logger.error("CONVERSION FAILED!")
        logger.error("Error type: %s", type(err_generic).__name__)
        logger.error("Error message: %s", err_generic)
        logger.exception("Full traceback:")
        logger.error("=" * 60)
        logger.error("Troubleshooting tips:")
        logger.error("  1. Check if S3 bucket is accessible")
        logger.error("  2. Verify AWS credentials are valid")
        logger.error(
            "  3. Try reducing target_block_size_mb if blocks are too large (currently %s MB)",
            target_block_size_mb,
        )
        logger.error("  4. Check available memory and disk space")
        logger.error("Failed after: %.1f seconds (%.2f minutes)", elapsed_seconds, elapsed_seconds / 60)
        logger.error("=" * 60)
        sys.stdout.flush()
        raise

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60

    if elapsed_hours >= 1:
        time_str = f"{elapsed_hours:.2f} hours ({elapsed_minutes:.1f} minutes)"
    elif elapsed_minutes >= 1:
        time_str = f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.1f} seconds)"
    else:
        time_str = f"{elapsed_seconds:.2f} seconds"

    logger.info("=" * 60)
    logger.info("MULTISCALE CONVERSION COMPLETED SUCCESSFULLY!")
    logger.info("Output written to: %s", output_zarr_path)
    logger.info("Total time: %s", time_str)
    logger.info("=" * 60)
    sys.stdout.flush()


def normalize_cluster_config_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    lower_value = str(path_value).strip().lower()
    if lower_value in {"none", "null", "false"}:
        return None
    normalized = Path(path_value).expanduser()
    try:
        normalized = normalized.resolve()
    except Exception:
        normalized = normalized.absolute()
    return str(normalized)


def execute_multiscale_job(
    *,
    input_zarr_path: str,
    output_zarr_path: str,
    chunk_size,
    voxel_size,
    n_lvls: int,
    scale_factor,
    target_block_size_mb: int,
    ray_num_cpus: int,
    use_ray: bool,
    ray_cluster_config_path: Optional[str],
) -> None:
    """Entry point mirroring affine_fusion_worker for cluster orchestration."""

    normalized_config_path = normalize_cluster_config_path(ray_cluster_config_path)

    if not normalized_config_path:
        perform_multiscale_conversion(
            input_zarr_path=input_zarr_path,
            output_zarr_path=output_zarr_path,
            use_ray=use_ray,
            ray_num_cpus=ray_num_cpus,
            chunk_size=chunk_size,
            voxel_size=voxel_size,
            n_lvls=n_lvls,
            scale_factor=scale_factor,
            target_block_size_mb=target_block_size_mb,
            remote_cluster=False,
        )
        return

    global ray_config_path, should_cleanup
    should_cleanup = False
    ray_config_path = normalized_config_path

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ray_config_dir = Path(normalized_config_path).parent
    yml_filename = Path(normalized_config_path).name

    try:
        cleanup_existing_cluster(yml_filename, ray_config_dir)
        start_cluster(yml_filename, ray_config_dir)

        multiscale_cmd = (
            "bash -lc \""
            "python3 - <<\\\"PY\\\"\n"
            "import sys\n"
            "sys.path.append('/home/ubuntu')\n"
            "from Rhapso.fusion.multiscale_worker import perform_multiscale_conversion\n"
            f"perform_multiscale_conversion(\n"
            f"    input_zarr_path={repr(input_zarr_path)},\n"
            f"    output_zarr_path={repr(output_zarr_path)},\n"
            f"    use_ray={repr(use_ray)},\n"
            f"    ray_num_cpus={ray_num_cpus},\n"
            f"    chunk_size={repr(chunk_size)},\n"
            f"    voxel_size={repr(voxel_size)},\n"
            f"    n_lvls={n_lvls},\n"
            f"    scale_factor={repr(scale_factor)},\n"
            f"    target_block_size_mb={target_block_size_mb},\n"
            "    remote_cluster=True,\n"
            ")\n"
            "PY\n"
            "\""
        )

        print("\n🔄 Starting multiscale conversion on cluster")
        print(f"   Input: {input_zarr_path}")
        print(f"   Output: {output_zarr_path}")

        subprocess.run(
            ["ray", "exec", yml_filename, multiscale_cmd],
            cwd=ray_config_dir,
            capture_output=False,
            text=True,
            check=True,
            timeout=8 * 60 * 60,  # 8 hour timeout
        )
    except subprocess.TimeoutExpired:
        print("❌ Multiscale conversion timed out after 8 hours")
        raise
    except subprocess.CalledProcessError as err_exec:
        print(f"❌ Multiscale conversion failed with exit code {err_exec.returncode}")
        raise
    finally:
        cleanup_cluster(yml_filename, ray_config_dir)


def run() -> None:
    """Convenience wrapper for running locally with default parameters."""
    execute_multiscale_job(
        input_zarr_path=DEFAULT_INPUT_ZARR_PATH,
        output_zarr_path=DEFAULT_OUTPUT_ZARR_PATH,
        chunk_size=DEFAULT_CHUNK_SIZE,
        voxel_size=DEFAULT_VOXEL_SIZE,
        n_lvls=DEFAULT_N_LEVELS,
        scale_factor=DEFAULT_SCALE_FACTOR,
        target_block_size_mb=DEFAULT_TARGET_BLOCK_SIZE_MB,
        ray_num_cpus=DEFAULT_RAY_NUM_CPUS,
        use_ray=True,
        ray_cluster_config_path=None,
    )


if __name__ == "__main__":
    input_zarr_path = DEFAULT_INPUT_ZARR_PATH
    output_zarr_path = DEFAULT_OUTPUT_ZARR_PATH
    chunk_size = DEFAULT_CHUNK_SIZE
    voxel_size = DEFAULT_VOXEL_SIZE
    n_lvls = DEFAULT_N_LEVELS
    scale_factor = DEFAULT_SCALE_FACTOR
    target_block_size_mb = DEFAULT_TARGET_BLOCK_SIZE_MB
    ray_num_cpus = DEFAULT_RAY_NUM_CPUS
    ray_cluster_config_path = DEFAULT_RAY_CONFIG_PATH

    print(f'{input_zarr_path=}')
    print(f'{output_zarr_path=}')
    print(f'{chunk_size=}')
    print(f'{voxel_size=}')
    print(f'{n_lvls=}')
    print(f'{scale_factor=}')
    print(f'{target_block_size_mb=}')
    print(f'{ray_num_cpus=}')
    print(f'{ray_cluster_config_path=}')

    execute_multiscale_job(
        input_zarr_path=input_zarr_path,
        output_zarr_path=output_zarr_path,
        chunk_size=chunk_size,
        voxel_size=voxel_size,
        n_lvls=n_lvls,
        scale_factor=scale_factor,
        target_block_size_mb=target_block_size_mb,
        ray_num_cpus=ray_num_cpus,
        use_ray=True,
        ray_cluster_config_path=ray_cluster_config_path,
    )
