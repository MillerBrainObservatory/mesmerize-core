"""
Useful functions adapted from old mesmerize

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""

import numpy as np
from functools import wraps
import os
from stat import S_IEXEC
from typing import *
import re as regex
from pathlib import Path
from warnings import warn
import sys
from tempfile import NamedTemporaryFile
from subprocess import check_call

import pandas as pd
import tifffile
import ffmpeg
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt

if os.name == "nt":
    IS_WINDOWS = True
    HOME = "USERPROFILE"
else:
    IS_WINDOWS = False
    HOME = "HOME"

if "MESMERIZE_LRU_CACHE" in os.environ.keys():
    MESMERIZE_LRU_CACHE = os.environ["MESMERIZE_LRU_CACHE"]
else:
    MESMERIZE_LRU_CACHE = 10


def save_mp4(fname: str | Path | np.ndarray, images, framerate=60, speedup=1, chunk_size=100, cmap="gray", win=7,
             vcodec='libx264'):
    """
    Save a video from a 3D array or TIFF stack to `.mp4`.

    Parameters
    ----------
    fname : str
        Output video file name.
    images : numpy.ndarray or str
        Input 3D array (T x H x W) or a file path to a TIFF stack.
    framerate : int, optional
        Original framerate of the video, by default 60.
    speedup : int, optional
        Factor to increase the playback speed, by default 1 (no speedup).
    chunk_size : int, optional
        Number of frames to process and write in a single chunk, by default 100.
    cmap : str, optional
        Colormap to apply to the video frames, by default "gray".
        Must be a valid Matplotlib colormap name.
    win : int, optional
        Temporal averaging window size. If `win > 1`, frames are averaged over
        the specified window using convolution. By default, 7.
    vcodec : str, optional
        Video codec to use, by default 'libx264'.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist when `images` is provided as a file path.
    ValueError
        If `images` is not a valid 3D NumPy array or a file path to a TIFF stack.

    Notes
    -----
    - The input array `images` must have the shape (T, H, W), where T is the number of frames,
      H is the height, and W is the width.
    - The `win` parameter performs temporal smoothing by averaging over adjacent frames.

    Examples
    --------
    Save a video from a 3D NumPy array with a colormap and speedup:

    >>> import numpy as np
    >>> images = np.random.rand(100, 600, 576) * 255
    >>> save_mp4('output.mp4', images, framerate=30, cmap='viridis', speedup=2)

    Save a video with temporal averaging applied over a 5-frame window at 4x speed:

    >>> save_mp4('output_smoothed.mp4', images, framerate=30, speedup=4, cmap='gray', win=5)

    Save a video from a TIFF stack:

    >>> save_mp4('output.mp4', 'path/to/stack.tiff', framerate=60, cmap='gray')
    """
    if isinstance(images, (str, Path)):
        if Path(images).is_file():
            images = tifffile.memmap(images)
        else:
            raise FileNotFoundError(f"File not found: {images}")

    T, height, width = images.shape
    colormap = cm.get_cmap(cmap)

    if win and win > 1:
        kernel = np.ones(win) / win
        images = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=0, arr=images)

    output_framerate = int(framerate * speedup)
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=output_framerate)
        .output(str(fname), pix_fmt='yuv420p', vcodec=vcodec, r=output_framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = images[start:end]
        colored_chunk = (colormap(chunk)[:, :, :, :3] * 255).astype(np.uint8)
        for frame in colored_chunk:
            process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()


def warning_experimental(more_info: str = ""):
    """
    decorator to warn the user that the function is experimental
    """

    def catcher(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            warn(
                f"You are trying to use the following experimental feature, "
                f"this may change in the future without warning:\n"
                f"{func.__qualname__}\n"
                f"{more_info}\n",
                FutureWarning,
                stacklevel=2
            )
            return func(self, *args, **kwargs)

        return fn

    return catcher


def validate_path(path: Union[str, Path]):
    if not regex.match("^[A-Za-z0-9@\/\\\:._-]*$", str(path)):
        raise ValueError(
            "Paths must only contain alphanumeric characters, "
            "hyphens ( - ), underscores ( _ ) or periods ( . )"
        )
    return path


def make_runfile(
        module_path: str, args_str: Optional[str] = None, filename: Optional[str] = None
) -> str:
    """
    Make an executable bash script.
    Used for running python scripts in external processes within the same python environment as the main/parent process.

    Parameters
    ----------
    module_path: str
        absolute path to the python module/script that should be run externally

    args_str: Optional[str]
        optinal str of args that is directly passed to the script specified by ``module_path``

    filename: Optional[str]
        optional, filename of the executable bash script

    Returns
    -------
    str
        path to the shell script that can be executed
    """

    if filename is None:
        if IS_WINDOWS:
            sh_file = os.path.join(os.environ[HOME], "run.ps1")
        else:
            sh_file = os.path.join(os.environ[HOME], "run.sh")
    else:
        if IS_WINDOWS:
            if not filename.endswith(".ps1"):
                filename = filename + ".ps1"

    sh_file = filename

    if args_str is None:
        args_str = ""

    if not IS_WINDOWS:
        with open(sh_file, "w") as f:

            f.write(f"#!/bin/bash\n")

            if "VIRTUAL_ENV" in os.environ.keys():
                f.write(
                    f'export PATH={os.environ["PATH"]}\n'
                    f'export VIRTUAL_ENV={os.environ["VIRTUAL_ENV"]}\n'
                    f'export LD_LIBRARY_PATH={os.environ["LD_LIBRARY_PATH"]}\n'
                )

            if "PYTHONPATH" in os.environ.keys():
                f.write(f'export PYTHONPATH={os.environ["PYTHONPATH"]}\n')

            # for k, v in os.environ.items():  # copy the current environment
            #     if '\n' in v:
            #         continue
            #
            # f.write(f'export {k}="{v}"\n')

            # User-setable n-processes
            if "MESMERIZE_N_PROCESSES" in os.environ.keys():
                f.write(
                    f'export MESMERIZE_N_PROCESSES={os.environ["MESMERIZE_N_PROCESSES"]}\n'
                )

            f.write(
                f"export OPENBLAS_NUM_THREADS=1\n"
                f"export MKL_NUM_THREADS=1\n"
            )

            if "CONDA_PREFIX" in os.environ.keys():
                # add command to run the python script in the conda environment
                # that was active at the time that this shell script was generated
                f.write(
                    f'{os.environ["CONDA_EXE"]} run -p {os.environ["CONDA_PREFIX"]} python {module_path} {args_str}')
            else:
                f.write(f"python {module_path} {args_str}")  # call the script to run

    else:
        with open(sh_file, "w") as f:
            for k, v in os.environ.items():  # copy the current environment
                if regex.match("^.*[\(\)]", str(k)) or regex.match("^.*[\(\)]", str(v)):
                    continue
                with NamedTemporaryFile(suffix=".ps1", delete=False) as tmp:
                    try:  # windows powershell is stupid so make sure all the env var names work
                        tmp.write(f'$env:{k}="{v}";\n')
                        tmp.close()
                        check_call(f"powershell {tmp.name}")
                        os.unlink(tmp.name)
                    except:
                        continue
                f.write(f'$env:{k}="{v}";\n')  # write only env vars that powershell likes
            f.write(f"{sys.executable} {module_path} {args_str}")

    st = os.stat(sh_file)
    os.chmod(sh_file, st.st_mode | S_IEXEC)

    print(sh_file)

    return sh_file


def quick_min_max(data: np.ndarray) -> Tuple[float, float]:
    # from pyqtgraph.ImageView
    # Estimate the min/max values of *data* by subsampling.
    # Returns [(min, max), ...] with one item per channel
    while data.size > 1e6:
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, 2)
        data = data[tuple(sl)]

    return float(np.nanmin(data)), float(np.nanmax(data))


def _organize_coordinates(contour: dict):
    coors = contour["coordinates"]
    coors = coors[~np.isnan(coors).any(axis=1)]

    return coors


def extract_center_square(images, size):
    """
    Extract a square crop from the center of the input images.

    Parameters
    ----------
    images : numpy.ndarray
        Input array. Can be 2D (H x W) or 3D (T x H x W), where:
        - H is the height of the image(s).
        - W is the width of the image(s).
        - T is the number of frames (if 3D).
    size : int
        The size of the square crop. The output will have dimensions
        (size x size) for 2D inputs or (T x size x size) for 3D inputs.

    Returns
    -------
    numpy.ndarray
        A square crop from the center of the input images. The returned array
        will have dimensions:
        - (size x size) if the input is 2D.
        - (T x size x size) if the input is 3D.

    Raises
    ------
    ValueError
        If `images` is not a NumPy array.
        If `images` is not 2D or 3D.
        If the specified `size` is larger than the height or width of the input images.

    Notes
    -----
    - For 2D arrays, the function extracts a square crop directly from the center.
    - For 3D arrays, the crop is applied uniformly across all frames (T).
    - If the input dimensions are smaller than the requested `size`, an error will be raised.

    Examples
    --------
    Extract a center square from a 2D image:

    >>> import numpy as np
    >>> image = np.random.rand(600, 576)
    >>> cropped = extract_center_square(image, size=200)
    >>> cropped.shape
    (200, 200)

    Extract a center square from a 3D stack of images:

    >>> stack = np.random.rand(100, 600, 576)
    >>> cropped_stack = extract_center_square(stack, size=200)
    >>> cropped_stack.shape
    (100, 200, 200)
    """
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    if images.ndim == 2:  # 2D array (H x W)
        height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[center_h - half_size:center_h + half_size,
               center_w - half_size:center_w + half_size]

    elif images.ndim == 3:  # 3D array (T x H x W)
        T, height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[:,
               center_h - half_size:center_h + half_size,
               center_w - half_size:center_w + half_size]
    else:
        raise ValueError("Input array must be 2D or 3D.")


def get_rand_mean_max(in_mov):
    rand = in_mov[np.random.randint(in_mov.shape[0])]
    mn = in_mov.mean(axis=0)
    mx = in_mov.max(axis=0)
    return rand, mn, mx


def plot_comparison(raw_mmap, reg_mmap, title=None, save_path=None):
    raw_rand, raw_mean, raw_max = get_rand_mean_max(raw_mmap)
    reg_rand, reg_mean, reg_max = get_rand_mean_max(reg_mmap)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    if title is not None:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    # Raw images
    axes[0, 0].imshow(raw_rand, cmap='gray')
    axes[0, 0].set_title("Raw - Random Frame")
    axes[0, 1].imshow(raw_mean, cmap='gray')
    axes[0, 1].set_title("Raw - Mean")
    axes[0, 2].imshow(raw_max, cmap='gray')
    axes[0, 2].set_title("Raw - Max")

    # Registered images
    axes[1, 0].imshow(reg_rand, cmap='gray')
    axes[1, 0].set_title("Registered - Random Frame")
    axes[1, 1].imshow(reg_mean, cmap='gray')
    axes[1, 1].set_title("Registered - Mean")
    axes[1, 2].imshow(reg_max, cmap='gray')
    axes[1, 2].set_title("Registered - Max")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def export_contours_with_params(row, save_path):
    params = row.params
    corr = row.caiman.get_corr_image()
    contours = row.cnmf.get_contours("good", swap_dim=False)[0]
    contours_bad = row.cnmf.get_contours("bad", swap_dim=False)[0]

    table_data = params["main"]
    df_table = pd.DataFrame(list(table_data.items()), columns=["Parameter", "Value"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(corr, cmap='gray')
    for contour in contours:
        axes[0].plot(contour[:, 0], contour[:, 1], color='cyan', linewidth=1)
    for contour in contours_bad:
        axes[0].plot(contour[:, 0], contour[:, 1], color='red', linewidth=0.2)

    axes[0].set_title(f'Accepted ({len(contours)}) and Rejected ({len(contours_bad)}) Neurons')
    axes[0].axis('off')
    axes[1].axis('tight')
    axes[1].axis('off')

    table = axes[1].table(cellText=df_table.values,
                          colLabels=df_table.columns,
                          loc='center',
                          cellLoc='center',
                          colWidths=[0.4, 0.6])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
