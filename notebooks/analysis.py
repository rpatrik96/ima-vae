import pandas as pd
import numpy as np


def sweep2df(sweep_runs, filename, save=False):
    data = []
    max_dim = -1
    for run in sweep_runs:

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        if run.state == "finished":

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config = {k: v for k, v in run.config.items() if not k.startswith("_")}

            dim = config["latent_dim"]

            mcc = summary["Metrics/val/mcc"]
            mcc_history = run.history(keys=[f"Metrics/val/mcc"])
            max_mcc_step, max_mcc = mcc_history.idxmax()[1], mcc_history.max()[1]

            cima = summary["Metrics/val/cima"]
            cima_history = run.history(keys=[f"Metrics/val/cima"])
            min_cima_step, min_cima = cima_history.idxmin()[1], cima_history.min()[1]

            mcc4min_cima = mcc_history.iloc[int(min_cima_step)]["Metrics/val/mcc"]
            cima4max_mcc = cima_history.iloc[int(max_mcc_step)]["Metrics/val/cima"]

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
            ).iloc[int(min_neg_elbo_step)]["Metrics/val/mse_obs_decoded_mean_latents"]

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
                col_norms_sq_history.iloc[int(min_neg_elbo_step)].to_numpy()[1:] ** 2
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