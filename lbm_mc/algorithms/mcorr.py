import traceback
import click
import caiman as cm
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import MotionCorrect
from caiman.summary_images import local_correlations_movie_offline
import matplotlib.pyplot as plt
import psutil
import os
from pathlib import Path, PurePosixPath
import numpy as np
from shutil import move as move_file
import time
import tifffile

# prevent circular import
if __name__ in ["__main__", "__mp_main__"]:  # when running in subprocess
    from lbm_mc import set_parent_raw_data_path, load_batch
    from lbm_mc.utils import save_mp4, extract_center_square, plot_comparison, get_rand_mean_max
else:  # when running with local backend
    from ..batch_utils import set_parent_raw_data_path, load_batch
    from ..utils import save_mp4, extract_center_square, plot_comparison, get_rand_mean_max

import matplotlib as mpl

mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (12, 8),
    'ytick.major.left': True,
})
jet = mpl.colormaps['jet']
jet.set_bad(color='k')


def run_algo(batch_path, uuid, data_path: str = None):
    algo_start = time.time()
    set_parent_raw_data_path(data_path)

    batch_path = Path(batch_path)
    df = load_batch(batch_path)

    item = df.caiman.uloc(uuid)
    # resolve full path
    input_movie_path = str(df.paths.resolve(item["input_movie_path"]))

    # because caiman doesn't let you specify filename to save memmap files
    # create dir with uuid as the dir item_name
    output_dir = Path(batch_path).parent.joinpath(str(uuid))
    output_summary_dir = Path(batch_path).parent.joinpath(".registration")
    output_summary_dir.mkdir(parents=False, exist_ok=True)

    caiman_temp_dir = str(output_dir)
    os.makedirs(caiman_temp_dir, exist_ok=True)
    os.environ["CAIMAN_TEMP"] = caiman_temp_dir
    os.environ["CAIMAN_NEW_TEMPFILE"] = "True"

    params = item["params"]

    # adapted from current demo notebook
    if "MESMERIZE_N_PROCESSES" in os.environ.keys():
        try:
            n_processes = int(os.environ["MESMERIZE_N_PROCESSES"])
        except:
            n_processes = psutil.cpu_count() - 1
    else:
        n_processes = psutil.cpu_count() - 1

    print("starting mc")
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend="local", n_processes=n_processes, single_thread=False
    )

    rel_params = dict(params["main"])
    opts = CNMFParams(params_dict=rel_params)
    # Run MC, denote boolean 'success' if MC completes w/out error
    try:
        # Run MC
        fnames = [input_movie_path]
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group("motion"))
        mc.motion_correct(save_movie=True)
        print(fnames)

        # find path to mmap file
        memmap_output_path_temp = df.paths.resolve(mc.mmap_file[0])

        # filename to move the output back to data dir
        mcorr_memmap_path = output_dir.joinpath(
            f"{uuid}-{memmap_output_path_temp.name}"
        )

        # move the output file
        move_file(memmap_output_path_temp, mcorr_memmap_path)

        print("mc finished successfully!")

        Yr, dims, T = cm.load_memmap(str(mcorr_memmap_path))
        images = np.reshape(Yr.T, [T] + list(dims), order="F")

        print("plotting raw-registered comparisons")
        plot_comparison(
            tifffile.memmap(input_movie_path),
            images,
            title=f"Raw vs Registered for uuid={uuid}",
            save_path=output_summary_dir.joinpath(f"{item.item_name}_mean_max_{uuid}.png"),
        )

        print("computing projections")
        proj_paths = dict()
        for proj_type in ["mean", "std", "max"]:
            p_img = getattr(np, f"nan{proj_type}")(images, axis=0)
            proj_paths[proj_type] = output_dir.joinpath(
                f"{uuid}_{proj_type}_projection.npy"
            )
            np.save(str(proj_paths[proj_type]), p_img)

        mp4_path = output_summary_dir / f"{item.item_name}_registered_{uuid}.mp4"
        print(f"Saving mp4 to {mp4_path}")

        # min-max normalize images
        norm_imgs = (images - images.min()) / (images.max() - images.min())
        # check if 256 is a valid size
        if norm_imgs.shape[1] < 256 or norm_imgs.shape[2] < 256:
            print("Image is too small to extract a 256x256 square")
            print("Using full image")
            save_mp4(str(mp4_path), norm_imgs)
        else:
            images = extract_center_square(norm_imgs, 256)
            save_mp4(str(mp4_path), images)

        np.save(str(proj_paths[proj_type]), p_img)
        print("Computing correlation image")
        Cns = local_correlations_movie_offline(
            str(mcorr_memmap_path),
            remove_baseline=True,
            window=1000,
            stride=1000,
            winSize_baseline=100,
            quantil_min_baseline=10,
            dview=dview,
        )
        Cn = Cns.max(axis=0)
        Cn[np.isnan(Cn)] = 0
        cn_path = output_dir.joinpath(f"{uuid}_cn.npy")
        np.save(str(cn_path), Cn, allow_pickle=False)
        cn_summary_path = output_summary_dir.joinpath(f"{item.item_name}_correlation_{uuid}.tiff")
        tifffile.imwrite(str(cn_summary_path), Cn)

        print("finished computing correlation image")

        # Compute shifts
        if opts.motion["pw_rigid"] == True:

            x_shifts = mc.x_shifts_els
            y_shifts = mc.y_shifts_els
            shifts = [x_shifts, y_shifts]
            if hasattr(mc, 'z_shifts_els'):
                shifts.append(mc.z_shifts_els)
            shift_path = output_dir.joinpath(f"{uuid}_shifts.npy")
            np.save(str(shift_path), shifts)

            # plot and save an image of shifts
            print("plotting shifts for pw_rigid=True")
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.plot(np.mean(shifts[0], axis=1), 'r')
            ax.plot(np.mean(shifts[1], axis=1), 'b')
            ax.set_title('Mean XY shifts per-patch (pw_rigid=True)')
            ax.set_xlabel('frames')
            ax.set_ylabel('pixels')
            ax.legend(['x shifts', 'y shifts'])
            plt.savefig(str(output_summary_dir.joinpath(f"{item.item_name}_reg_shifts_{uuid}.png")))
        else:
            shifts = mc.shifts_rig
            shift_path = output_dir.joinpath(f"{uuid}_shifts.npy")
            np.save(str(shift_path), shifts)

            # for rigid, transpose and covert to list
            shifts = list(np.array(shifts).T)
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.plot(shifts[0], 'r')
            ax.plot(shifts[1], 'b')
            ax.set_title('Rigid XY shifts (pw_rigid=False)')
            ax.set_xlabel('frames')
            ax.set_ylabel('pixels')
            ax.legend(['x shifts', 'y shifts'])
            plt.savefig(str(output_summary_dir.joinpath(f"{item.item_name}_shifts_{uuid}.png")))

        # output dict for pandas series for dataframe row
        d = dict()

        # save paths as relative path strings with forward slashes
        cn_path = str(PurePosixPath(cn_path.relative_to(output_dir.parent)))
        mcorr_memmap_path = str(PurePosixPath(mcorr_memmap_path.relative_to(output_dir.parent)))
        shift_path = str(PurePosixPath(shift_path.relative_to(output_dir.parent)))
        results_path = PurePosixPath(output_summary_dir.relative_to(output_dir.parent))
        for proj_type in proj_paths.keys():
            d[f"{proj_type}-projection-path"] = str(PurePosixPath(proj_paths[proj_type].relative_to(
                output_dir.parent
            )))

        d.update(
            {
                "mcorr-output-path": mcorr_memmap_path,
                "results-path": results_path,
                "seg-path": results_path / '.segmentation',
                "reg-path": results_path / '.registration',
                "corr-img-path": cn_path,
                "shifts": shift_path,
                "success": True,
                "traceback": None,
            }
        )

    except:
        d = {"success": False, "traceback": traceback.format_exc()}
        print("mc failed, stored traceback in output")

    cm.stop_server(dview=dview)

    runtime = round(time.time() - algo_start, 2)
    df.caiman.update_item_with_results(uuid, d, runtime)


@click.command()
@click.option("--batch-path", type=str)
@click.option("--uuid", type=str)
@click.option("--data-path", type=str)
def main(batch_path, uuid, data_path: str = None):
    run_algo(batch_path, uuid, data_path)


if __name__ == "__main__":
    main()
