import pandas as pd
import numpy as np
from os.path import isfile


def sweep2df(sweep_runs, filename, save=False, load=False):
    if load is True and isfile(filename) is True:
        print(f"\t Loading {filename}...")
        return pd.read_csv(filename)
    data = []
    max_dim = -1
    for run in sweep_runs:

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        if run.state == "finished":
            try:

                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config = {k: v for k, v in run.config.items() if not k.startswith("_")}

                dim = config["latent_dim"]

                mcc = summary["Metrics/val/mcc"]
                mcc_history = run.history(keys=[f"Metrics/val/mcc"])
                max_mcc_step, max_mcc = mcc_history.idxmax()[1], mcc_history.max()[1]

                cima = summary["Metrics/val/cima"]
                cima_history = run.history(keys=[f"Metrics/val/cima"])
                min_cima_step, min_cima = (
                    cima_history.idxmin()[1],
                    cima_history.min()[1],
                )

                mcc4min_cima = mcc_history.iloc[int(min_cima_step)]["Metrics/val/mcc"]
                cima4max_mcc = cima_history.iloc[int(max_mcc_step)]["Metrics/val/cima"]

                mixing_linear_map_cima = (
                    -1
                    if "mixing_linear_map_cima" not in summary.keys()
                    else summary["mixing_linear_map_cima"]
                )
                mixing_cima = (
                    -1
                    if "mixing_cima" not in summary.keys()
                    else summary["mixing_cima"]
                )

                gamma_square = 1.0 / config["decoder_var"]
                prior = config["prior"]

                neg_elbo = summary["Metrics/val/neg_elbo"]
                neg_elbo_history = run.history(keys=[f"Metrics/val/neg_elbo"])
                min_neg_elbo_step, min_neg_elbo = (
                    neg_elbo_history.idxmin()[1],
                    neg_elbo_history.min()[1],
                )

                mse_sources_mean_decoded_sources = run.history(
                    keys=[f"Metrics/val/mse_sources_mean_decoded_sources"]
                ).iloc[int(min_neg_elbo_step)][
                    "Metrics/val/mse_sources_mean_decoded_sources"
                ]
                mse_obs_decoded_mean_latents = run.history(
                    keys=[f"Metrics/val/mse_obs_decoded_mean_latents"]
                ).iloc[int(min_neg_elbo_step)][
                    "Metrics/val/mse_obs_decoded_mean_latents"
                ]

                sigmas_history = run.history(
                    keys=[
                        f"Metrics/val/latent_statistics.latent_variance_{i}"
                        for i in range(dim)
                    ]
                )
                col_norms_sq_history = run.history(
                    keys=[f"Metrics/val/col_norm_{i}" for i in range(dim)]
                )
                sigmas = sigmas_history.iloc[int(min_neg_elbo_step)].to_numpy()[1:]
                col_norms_sq = (
                    col_norms_sq_history.iloc[int(min_neg_elbo_step)].to_numpy()[1:]
                    ** 2
                )
                # [run.summary._json_dict[f"Metrics/val/col_norm_{i}"]**2 for i in range(dim)]

                mcc4min_neg_elbo = mcc_history.iloc[int(min_neg_elbo_step)][
                    "Metrics/val/mcc"
                ]
                cima4min_neg_elbo = cima_history.iloc[int(min_neg_elbo_step)][
                    "Metrics/val/cima"
                ]

                rec_loss = summary["Metrics/val/rec_loss"]
                kl_loss = summary["Metrics/val/kl_loss"]
                rhs = 1.0 / (
                    float(prior == "gaussian") + gamma_square * np.array(col_norms_sq)
                )
                if dim > max_dim:
                    max_dim = dim

                data.append(
                    [
                        run.name,
                        dim,
                        gamma_square,
                        neg_elbo,
                        kl_loss,
                        rec_loss,
                        prior,
                        cima,
                        mcc,
                        min_cima,
                        min_cima_step,
                        cima4max_mcc,
                        cima4min_neg_elbo,
                        mixing_linear_map_cima,
                        mixing_cima,
                        max_mcc,
                        max_mcc_step,
                        mcc4min_cima,
                        mcc4min_neg_elbo,
                        min_neg_elbo,
                        min_neg_elbo_step,
                        mse_sources_mean_decoded_sources,
                        mse_obs_decoded_mean_latents,
                        *sigmas,
                        *col_norms_sq,
                        *rhs,
                    ]
                )
            except:
                print(f"Encountered a faulty run with ID {run.name}")

    runs_df = pd.DataFrame(
        data,
        columns=[
            "name",
            "dim",
            "gamma_square",
            "neg_elbo",
            "kl_loss",
            "rec_loss",
            "prior",
            "cima",
            "mcc",
            "min_cima",
            "min_cima_step",
            "cima4max_mcc",
            "cima4min_neg_elbo",
            "mixing_linear_map_cima",
            "mixing_cima",
            "max_mcc",
            "max_mcc_step",
            "mcc4min_cima",
            "mcc4min_neg_elbo",
            "min_neg_elbo",
            "min_neg_elbo_step",
            "mse_sources_mean_decoded_sources",
            "mse_obs_decoded_mean_latents",
            *[f"sigma_{i}" for i in range(max_dim)],
            *[f"col_norm_sq_{i}" for i in range(max_dim)],
            *[f"rhs_{i}" for i in range(max_dim)],
        ],
    ).fillna(0)

    if save is True:
        runs_df.to_csv(filename)

    return runs_df


def format_violin(vp, facecolor="#1A85FF"):
    for el in vp["bodies"]:
        el.set_facecolor(facecolor)
        el.set_edgecolor("black")
        el.set_linewidth(1.5)
        el.set_alpha(0.9)
    for pn in ["cbars", "cmins", "cmaxes", "cmedians"]:
        vp_ = vp[pn]
        vp_.set_edgecolor("black")
        vp_.set_linewidth(1)


import matplotlib.pyplot as plt


def create_violinplot(groups, xlabel, ylabel, xticklabels, filename=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax = ax.twinx()

    vp = ax.violinplot(groups, showmedians=True)
    format_violin(vp, "#1A85FF")

    ax.set_xticklabels(xticklabels)
    # ax.set_xticks(xticks)
    # plt.locator_params(axis='y', nbins=5)
    # plt.yticks(fontsize=24)
    # plt.ylim([0, 0.5])
    ax.set_ylabel(ylabel)
    # ax.set_xlabel(xlabel)
    if filename is not None:
        plt.savefig(f"{filename}.svg")
    return ax


def violin_by_prior(
    gauss_data,
    laplace_data,
    uniform_data,
    xticks,
    xlabel,
    ylabel,
    offset,
    filename,
    figsize=(8, 6),
    log=False,
):

    plt.figure(figsize=figsize)
    vp_gauss = plt.violinplot(
        [np.log10(i) if log is True else i for i in gauss_data], positions=xticks
    )
    vp_laplace = plt.violinplot(
        [np.log10(i) if log is True else i for i in laplace_data],
        positions=-offset + xticks,
    )
    vp_uniform = plt.violinplot(
        [np.log10(i) if log is True else i for i in uniform_data],
        positions=offset + xticks,
    )
    plt.legend(
        [vp_gauss["bodies"][0], vp_laplace["bodies"][0], vp_uniform["bodies"][0]],
        ["gaussian", "laplace", "uniform"],
        loc="upper right",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    # plt.tight_layout()
    plt.savefig(filename)
