# %%

from pathlib import Path

import einops
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import os

from utils.render import colorize, make_Rt, render_point_clouds

methods = [
    "Neighbor_interpolation",
    "Bilinear_interpolation",
    "Bicubic_interpolation",
    "LIIF",
    "LSR",
    "ILN",
    # "diffusion_lidargen_fixed",
    "R2DM",
    "Ours",
]


def render_xyz(xyz, max_depth=80.0):
    z_min, z_max = -2 / max_depth, 0.5 / max_depth
    z = (xyz[:, [2]] - z_min) / (z_max - z_min)
    colors = colorize(z.clamp(0, 1), cm.viridis) / 255
    points = einops.rearrange(xyz, "B C H W -> B (H W) C") / max_depth
    colors = 1 - einops.rearrange(colors, "B C H W -> B (H W) C")
    R, t = make_Rt(pitch=torch.pi / 4, yaw=torch.pi / 4, z=0.6, device=xyz.device)
    bev = 1 - render_point_clouds(points=points, colors=colors, R=R, t=t)
    bev = einops.rearrange(bev, "B C H W -> B H W C")
    return bev


def redner_img(img):
    img = colorize(img)
    img = einops.rearrange(img, "B C H W -> B H W C")
    return img


def parse_data(index, root):
    root = Path(root)

    elements = check_common_files(methods)

    # prediction
    for method in methods:
        # method_dir = root / method
        # pred_path = sorted(method_dir.glob("results/*.pth"))[index]
        pred_path = root / method / "results" / elements[index]

        tensor = torch.load(pred_path, map_location="cpu")  # (5,H,W)
        d, xyz, r = tensor.split([1, 3, 1], dim=0)
        yield method, d, r, xyz
    # ground truth
    # gt_path = str(pred_path).replace("results", "targets")
    # tensor = torch.load(gt_path, map_location="cpu")  # (5,H,W)
    # d, xyz, r = tensor.split([1, 3, 1], dim=0)
    # yield "gt", d, r, xyz


def check_common_files(paths):
    ext_paths = [
        f"baseline_results/baseline_results/{paths[i]}/results/"
        for i in range(len(paths))
    ]
    files = [os.listdir(path) for path in ext_paths]

    elements_in_all = list(set.intersection(*map(set, files)))

    return elements_in_all


def main():
    # sample_index = 1
    # packed = parse_data(index=sample_index, root="baseline_results/baseline_results")
    # methods, Ds, Rs, XYZs = zip(*packed)

    # Ds = torch.stack(Ds)
    # Rs = torch.stack(Rs)
    # XYZs = torch.stack(XYZs)

    # Ds = redner_img(Ds / 80.0)  # (B,H,W,3)
    # Rs = redner_img(Rs)  # (B,H,W,3)
    # BEVs = render_xyz(XYZs)  # (B,H,W,3)

    # fig, ax = plt.subplots(
    #     3,
    #     len(methods),
    #     figsize=(20, 5),
    #     gridspec_kw={"height_ratios": [100, 30, 30]},
    #     constrained_layout=True,
    # )
    # for i, method in enumerate(methods):
    #     ax[0][i].set_title(method)
    #     ax[0][i].imshow(BEVs[i])
    #     ax[1][i].imshow(Ds[i, :, 256 * 2 : 256 * 3], interpolation="none")
    #     ax[2][i].imshow(Rs[i, :, 256 * 2 : 256 * 3], interpolation="none")
    #     plt.savefig(
    #         "baseline_comparison.pdf", bbox_inches="tight", pad_inches=0, dpi=500
    #     )
    # [a.axis("off") for a in ax.ravel()]
    # plt.show()

    sample_index = 50
    packed = parse_data(index=sample_index, root="baseline_results/baseline_results")
    methods, Ds, Rs, XYZs = zip(*packed)

    Ds = torch.stack(Ds)
    Rs = torch.stack(Rs)
    XYZs = torch.stack(XYZs)

    Ds = redner_img(Ds / 80.0)  # (B,H,W,3)
    Rs = redner_img(Rs)  # (B,H,W,3)
    BEVs = render_xyz(XYZs)  # (B,H,W,3)

    fig, ax = plt.subplots(
        6,
        len(methods) // 2,
        figsize=(20, 15),
        gridspec_kw={"height_ratios": [100, 30, 30, 100, 30, 30]},
        constrained_layout=True,
    )

    for i in range(len(methods) // 2):
        ax[0][i].set_title(methods[i].replace("_", " "), fontsize=30)
        ax[0][i].imshow(BEVs[i])
        ax[1][i].imshow(Ds[i, :, 256 * 2 : 256 * 3], interpolation="none")
        ax[2][i].imshow(Rs[i, :, 256 * 2 : 256 * 3], interpolation="none")
        ax[3][i].set_title(
            methods[i + (len(methods) // 2)].replace("_", " "), fontsize=30
        )
        ax[3][i].imshow(BEVs[i + (len(methods) // 2)])
        ax[4][i].imshow(
            Ds[i + (len(methods) // 2), :, 256 * 2 : 256 * 3], interpolation="none"
        )
        ax[5][i].imshow(
            Rs[i + (len(methods) // 2), :, 256 * 2 : 256 * 3], interpolation="none"
        )

    [a.axis("off") for a in ax.ravel()]
    plt.savefig("baseline_comparison.pdf", bbox_inches="tight", pad_inches=0, dpi=500)
    plt.show()


if __name__ == "__main__":
    main()

# %%
