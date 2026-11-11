import pandas as pd 
import numpy as np 
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import time
from wassersteinDL import wasserstein_DL
import warnings
import os
from sklearn.manifold import MDS
from pathlib import Path
import argparse
import sys
import datetime
from config import *

# use scikit-learn for the baseline KLNMF
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from scipy.optimize import linear_sum_assignment, nnls

outputdir = "./tmp_output/sig_plots"
os.system(f"mkdir -p {outputdir}")

import datetime

now = datetime.datetime.now()
cosmic = pd.read_csv(COSMIC_FILE, sep="\t", index_col=0)
sbs96 = cosmic.index
all_cosmic_sigs = cosmic.columns

def run_simulation(k = 10, N = 200, r = 50):
    # cosmic.head()
    # select some SBS from the COSMIC catalog to simulate the data. 

    # number of signatures from COSMIC
    # k = 10

    # number of samples in the dataset
    # N = 200

     # NB_DISPERSION
    # r = 50 
    
    cosmic_sigs = random.sample(list(all_cosmic_sigs), k)

    print(f"Selected COSMIC signature to do simulation: {cosmic_sigs}")

    W_true = cosmic[cosmic_sigs].to_numpy()
    rng = np.random.default_rng(SEED)
    H_prop = np.zeros((k, N))
    for j in range(N):
        active = rng.random(k) < 0.5
        active[0] = True # first SBS on the list is always on
        H_prop[active, j] = rng.dirichlet(np.ones(active.sum()))
    burden = rng.lognormal(np.log(1000), 0.7, N)

    # multiply the proportion, tumor burden and the signatures to get simulated real counts
    mu = W_true @ (H_prop * burden)

    # the final SBS 96 x N matrix for running NMF - KLNMF - WNMF
    X = rng.negative_binomial(r, r / (r + mu))

    H_true = H_prop * burden  # true exposures, in mutation counts

    return X, W_true, H_true, cosmic_sigs

def read_input(path_to_X):
    # read a 96 x N count matrix (rows = SBS96 channels), reorder rows to match sbs96
    X = pd.read_csv(path_to_X, index_col=0)
    if not set(sbs96) <= set(X.index) and set(sbs96) <= set(X.columns):
        X = X.T  # samples x 96 -> 96 x samples
    X = X.loc[sbs96].apply(pd.to_numeric, errors="raise")
    X = X.loc[:, X.sum(axis=0) > 0]  # drop samples with no mutations
    return X.to_numpy(float), list(X.columns)

def read_cost_matrix(path_to_M):
    # read the saved cost matrix, but make sure that the order is the same as sbs96
    M = (pd.read_csv(path_to_M, index_col=0).loc[sbs96, sbs96].to_numpy(float))
    return M


# ***** HELPER FUNCTIONS FOR THE VALIDATION MODULE *****
# This cell contains the VALIDATION module

def cosine_similarity(A, B):
    """Calculate the cosine similartiy between the columns of A (n x p) and B (n x q) -> p x q."""
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    return A.T @ B


def pam(Dist, k, max_iter=100):
    """PAM / k-medoids on a distance matrix: greedy BUILD start, then SWAP."""
    n = len(Dist)
    medoids = [int(np.argmin(Dist.sum(1)))]
    while len(medoids) < k:
        nearest = Dist[:, medoids].min(1)
        gain = np.maximum(nearest[:, None] - Dist, 0).sum(0)
        gain[medoids] = -np.inf
        medoids.append(int(np.argmax(gain)))
    medoids = np.array(medoids)
    cost = Dist[:, medoids].min(1).sum()
    for _ in range(max_iter):
        best = (cost, None, None)
        for mi in range(k):
            for h in range(n):
                if h in medoids:
                    continue
                trial = medoids.copy(); trial[mi] = h
                c = Dist[:, trial].min(1).sum()
                if c < best[0] - 1e-12:
                    best = (c, mi, h)
        if best[1] is None:
            break
        cost, medoids[best[1]] = best[0], best[2]
    return medoids, np.argmin(Dist[:, medoids], axis=1)


def consensus_signatures(runs, k):
    """Step 1-2: pool the signatures of all runs, cluster with PAM on cosine distance."""
    pooled = np.hstack(runs)                                  # 96 x (R*k)
    Dist = np.clip(1 - cosine_similarity(pooled, pooled), 0, None)
    np.fill_diagonal(Dist, 0)
    med, labels = pam(Dist, k)
    medoids = pooled[:, med] / pooled[:, med].sum(0)
    cohesion = np.array([1 - Dist[labels == c, med[c]].mean() for c in range(k)])
    return medoids, cohesion


def match_signatures(W_true, W_est):
    """Step 3: one-to-one Hungarian matching on cosine."""
    cs = cosine_similarity(W_true, W_est)
    r, c = linear_sum_assignment(-cs)
    return c, cs[r, c]                                        # est index per true sig, cosine


def confusion_at_cutoff(match_cos, n_est, t):
    """Step 4: TP / FP / FN at cosine cutoff t."""
    tp = int((match_cos >= t).sum())
    fp, fn = n_est - tp, len(match_cos) - tp
    prec, rec = tp / n_est, tp / len(match_cos)
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return dict(cutoff=t, TP=tp, FP=fp, FN=fn, precision=prec, TPR=rec, F1=f1)


def validate(W_true, runs, cutoffs=np.round(np.arange(0.70, 1.0001, 0.01), 2)):
    """Full section 3.3 validation: medoids, matches and the TPR curve (step 5)."""
    k = W_true.shape[1]
    medoids, cohesion = consensus_signatures(runs, k)
    idx, cos = match_signatures(W_true, medoids)
    curve = [confusion_at_cutoff(cos, medoids.shape[1], t) for t in cutoffs]
    return medoids, idx, cos, cohesion[idx], curve

def unit(A):
    return A / np.linalg.norm(A, axis=0, keepdims=True)

# ***** HELPER FUNCTIONS FOR THE REFIT AFTER N RUNS *****
from wassersteinDL.wasserstein_DL import checkOptions, createHandlers

def refit_wnmf(X, S, M, gamma, rhoL, rhoD, options):
    # Refit the WNMF model using the chosen medoid. 
    P = X / X.sum(0, keepdims=True)
    opt = checkOptions(options)
    optimizeLambda, _, _, sizeD = createHandlers(opt, P, M, S.shape[1], gamma, rhoL, rhoD)
    lam, _, _ = optimizeLambda(S, np.zeros(sizeD))
    return lam * X.sum(0)                                   # proportions -> mutations

def sk_klnmf(X, k, init="random", seed=None, max_iter=5000, tol=1e-6):
    # One sklearn KL-NMF fit; W columns rescaled to the simplex, H in counts.
    X = np.asarray(X, float)
    model = NMF(n_components=k, beta_loss="kullback-leibler", solver="mu", init=init,
                max_iter=max_iter, tol=tol, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W = model.fit_transform(X)
    H = model.components_.copy()
    s = W.sum(0, keepdims=True) + 1e-12
    return dict(W=W / s, H=H * s.T)

def refit_klnmf(X, S, n_iter=2000):
    # Refit the KLNMF model using the chosen medoid. 
    # KL-NMF weights with the signatures S held fixed: the H update of KL-NMF alone
    X = np.asarray(X, float)
    H = np.ones((S.shape[1], X.shape[1])) * X.sum(0) / S.shape[1]
    for _ in range(n_iter):
        H *= (S.T @ (X / (S @ H + 1e-12))) / S.sum(0)[:, None]
    return H


# ***** HELPER FUNCTIONS FOR PLOTTING *****
SUB_COLORS = {"C>A": "#1EBFF0",
              "C>G": "#050708",
              "C>T": "#E62725",
              "T>A": "#CBCACB",
              "T>C": "#A1CF64",
              "T>G": "#EDC8C5"}

def plot_mds(sig_runs_by_method, results, path, W_true=None, true_names=None):
    # MDS map of the run signatures (dots), the medoids (stars) and, if known, the true signatures (x)
    fig, axes = plt.subplots(1, len(sig_runs_by_method), figsize=(6 * len(sig_runs_by_method), 5),
                             squeeze=False)
    for ax, (name, sig_runs) in zip(axes[0], sig_runs_by_method.items()):
        pooled = np.hstack(sig_runs)
        med = results[name]["medoids"]
        labels = np.argmax(unit(pooled).T @ unit(med), axis=1)
        allsig = np.hstack([pooled, med] + ([W_true] if W_true is not None else []))
        D = np.clip(1 - unit(allsig).T @ unit(allsig), 0, None)
        np.fill_diagonal(D, 0)
        D = (D + D.T) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            xy = MDS(n_components=2, dissimilarity="precomputed", random_state=0).fit_transform(D)
        n_p, n_m = pooled.shape[1], med.shape[1]
        p, m = xy[:n_p], xy[n_p:n_p + n_m]
        for c in range(n_m):
            color = plt.cm.tab20(c % 20)
            ax.scatter(*p[labels == c].T, s=25, alpha=0.6, color=color)
            ax.scatter(*m[c], marker="*", s=300, color=color, edgecolor="k")
            ax.annotate(results[name]["names"][c], m[c], xytext=(5, -10),
                        textcoords="offset points", fontsize=7)
        title = f"{name}: run signatures (dots), medoids (stars)"
        if W_true is not None:
            t = xy[n_p + n_m:]
            ax.scatter(*t.T, marker="x", s=80, color="k")
            for i, s in enumerate(true_names):
                ax.annotate(s, t[i], xytext=(5, 5), textcoords="offset points", fontsize=9)
            title += ", true (x)"
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_sbs(panels, path):
    # stack of SBS96 bar plots sharing one y scale. panels: list of (title, 96-vector in sbs96 order)
    order = sorted(range(96), key=lambda r: (sbs96[r][2:5], sbs96[r][0], sbs96[r][-1]))
    bar_colors = [SUB_COLORS[sbs96[r][2:5]] for r in order]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.2 * len(panels)), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    for ax, (title, vec) in zip(axes, panels):
        ax.bar(range(96), np.asarray(vec)[order], color=bar_colors, width=0.8)
        ax.set_title(title, loc="left")
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax); ax.set_ylabel("probability")
    axes[-1].set_xticks(range(0, 96, 16)); axes[-1].set_xticklabels(list(SUB_COLORS), ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ***** MAIN RUN *****
def add_arguments(parser):
    # input data: if --input is not given, X is simulated from COSMIC signatures
    g = parser.add_argument_group("input data")
    g.add_argument("--input", "-i", default=None,
                   help="Path to a 96 x N count matrix (.csv, SBS96 channels as row index). "
                        "If omitted, X is simulated from COSMIC signatures.")
    g.add_argument("--cost-matrix", "-M", required=True,
                   help="Path to the 96 x 96 ground cost matrix (.csv) with SBS96 labels, "
                        "e.g. cost_matrix/M1.csv")
    g.add_argument("--k", "-k", type=int, default=10,
                   help="Number of signatures to extract (and to simulate, if no --input)")
    g.add_argument("--outdir", "-o", default=outputdir, help="Output folder")
    g.add_argument("--runName", "-n", default=f"{now.strftime('%Y%m%d')}", help="Run name")

    # simulation (only used when --input is not given)
    s = parser.add_argument_group("simulation (only used without --input)")
    s.add_argument("--N", type=int, default=200, help="Number of simulated samples")
    s.add_argument("--r", type=float, default=50, help="Negative-binomial dispersion")

    # model / validation
    m = parser.add_argument_group("W-NMF / validation")
    m.add_argument("--N-RUNS", dest="N_RUNS", type=int, default=10,
                   help="Number of independent runs for W-NMF and KL-NMF")
    m.add_argument("--gamma", type=float, default=1 / 50, help="Entropic regularisation")
    m.add_argument("--rhoL", type=float, default=0.05, help="Entropy weight on lambda")
    m.add_argument("--rhoD", type=float, default=0.05, help="Entropy weight on D")
    m.add_argument("--stop", type=float, default=1e-3, help="Outer stopping tolerance")
    m.add_argument("--D-step-stop", dest="D_step_stop", type=float, default=5e-5)
    m.add_argument("--lambda-step-stop", dest="lambda_step_stop", type=float, default=5e-4)
    m.add_argument("--alpha", type=float, default=0.5)
    m.add_argument("--verbose", type=int, default=0)
    return parser

def run(args):
    k = args.k
    N_RUNS = args.N_RUNS
    path_to_save_output = os.path.join(args.outdir, args.runName)
    os.makedirs(path_to_save_output, exist_ok=True)

    # ***** input the matrix X, read the path if given otherwise simulate data
    if args.input is None:
        
        print(" ***** ***** ***** ***** ***** ***** ***** *****")
        print(" ***** NO INPUT PATH IS GIVEN *****")
        print(" ***** USING THE SIMULATION DATA *****")
        print(" ***** ***** ***** ***** ***** ***** ***** *****")

        X, W_true, H_true, cosmic_sigs = run_simulation(k=k, N=args.N, r=args.r)
        sample_names = [f"Sample{j + 1}" for j in range(X.shape[1])]
        pd.DataFrame(X, index=sbs96, columns=sample_names).to_csv(f"{path_to_save_output}/X_simulated.csv")
        pd.DataFrame(W_true, index=sbs96, columns=cosmic_sigs).to_csv(f"{path_to_save_output}/W_true.csv")
        pd.DataFrame(H_true, index=cosmic_sigs, columns=sample_names).to_csv(f"{path_to_save_output}/H_true.csv")
    else:
        print("***** reading in the input data from *****")
        print(args.input)
        X, sample_names = read_input(args.input)
        W_true, H_true, cosmic_sigs = None, None, None  # no ground truth for real data
    
    print(f"X: {X.shape[0]} channels x {X.shape[1]} samples")

    M = read_cost_matrix(args.cost_matrix)
    
    gamma, rhoL, rhoD = args.gamma, args.rhoL, args.rhoD
    options = {"stop": args.stop,
               "D_step_stop": args.D_step_stop,
               "lambda_step_stop": args.lambda_step_stop,
               "alpha": args.alpha,
               "Kmultiplication": "symmetric",
               "verbose": args.verbose}

    # ***** Run W-NMF and KL-NMF N_RUNS times each
    P = X / X.sum(0, keepdims=True) # make sure everything is histogram.

    # W-NMF: one fit per seed, keep the dictionary D (= S, 96 x k).
    runs_wnmf = []
    t = time.time()
    for r in range(N_RUNS):
        S_r = wasserstein_DL(P, k, M, gamma, rhoL, rhoD, options,
                             rng=np.random.default_rng(100 + r))[0]
        runs_wnmf.append(S_r)
        print(f"W-NMF run {r + 1}/{N_RUNS} done ({time.time() - t:.0f}s)")

    # KL-NMF: one random-init sklearn fit per seed. W columns are on the simplex.
    runs_klnmf = [sk_klnmf(X, k, init="random", seed=100 + r)["W"] for r in range(N_RUNS)]

    # ***** Consensus signatures; validation (section 3.3) only when the truth is known
    results = {}
    for name, sig_runs in [("W-NMF", runs_wnmf), ("KL-NMF", runs_klnmf)]:
        tag = name.replace("-", "")
        if W_true is None:
            medoids, cohesion = consensus_signatures(sig_runs, k)
            names = [f"{tag}_sig{c + 1}" for c in range(k)]
            results[name] = dict(medoids=medoids, cohesion=cohesion, names=names)
        else:
            medoids, idx, cos, cohesion, curve = validate(W_true, sig_runs)
            # reorder the medoids so that column i is the one matched to the true signature i
            medoids = medoids[:, idx]
            names = [f"{tag}_{s}" for s in cosmic_sigs]
            curve = pd.DataFrame(curve).set_index("cutoff")
            results[name] = dict(medoids=medoids, cos=cos, cohesion=cohesion, curve=curve, names=names)
            print(f"{name} cosine per true signature:", dict(zip(cosmic_sigs, cos.round(3))))
        print(f"{name} cluster cohesion:", cohesion.round(3))

    if W_true is not None:
        plt.figure(figsize=(5, 3.5))
        for name, res in results.items():
            plt.plot(res["curve"].index, res["curve"].TPR, marker="o", ms=3, label=name)
        plt.xlabel("cosine cutoff"); plt.ylabel("TPR (= precision = recall)")
        plt.title(f"Recovery of {k} true signatures ({N_RUNS} runs)"); plt.legend()
        plt.tight_layout(); plt.savefig(f"{path_to_save_output}/TPR_curve.png", dpi=150); plt.close()
        pd.DataFrame({name: res["cos"] for name, res in results.items()},
                     index=cosmic_sigs).to_csv(f"{path_to_save_output}/cosine_to_true.csv")

    # ***** Refit to get the weight after getting the consensus signatures after N_RUNS
    refit = {"W-NMF": lambda S: refit_wnmf(X, S, M, gamma, rhoL, rhoD, options),
             "KL-NMF": lambda S: refit_klnmf(X, S)}
    expo_eval = {}
    for name, res in results.items():
        tag = name.replace("-", "")
        H_est = refit[name](res["medoids"])
        res["exposures"] = H_est
        # save the signature and weight matrix
        pd.DataFrame(res["medoids"], index=sbs96, columns=res["names"]
                     ).to_csv(f"{path_to_save_output}/{tag}_signatures.csv")
        pd.DataFrame(H_est, index=res["names"], columns=sample_names
                     ).to_csv(f"{path_to_save_output}/{tag}_exposures.csv")
        if H_true is not None:
            corr = [np.corrcoef(H_true[i], H_est[i])[0, 1] for i in range(k)]
            rel_err = np.abs(H_est - H_true).sum(0) / H_true.sum(0)
            expo_eval[name] = dict(zip(cosmic_sigs, corr), median_rel_err=np.median(rel_err))
    if expo_eval:
        expo_eval = pd.DataFrame(expo_eval).T
        print("Exposure correlation with the truth:\n", expo_eval.round(3))
        expo_eval.to_csv(f"{path_to_save_output}/exposure_eval.csv")

    # ***** Plot MDS to see how N_RUNS signature are
    plot_mds({"W-NMF": runs_wnmf, "KL-NMF": runs_klnmf}, results,
             f"{path_to_save_output}/MDS_signatures.png", W_true=W_true, true_names=cosmic_sigs)

    # ***** SBS96 profile plots, one file per signature
    sbs_dir = f"{path_to_save_output}/sbs_plots"
    os.makedirs(sbs_dir, exist_ok=True)
    wn, kl = results["W-NMF"], results["KL-NMF"]
    if W_true is not None:
        # true signature on top, then the medoid each method matched to it
        for i, sig in enumerate(cosmic_sigs):
            panels = [(f"{sig}: true (COSMIC)", W_true[:, i])]
            panels += [(f"{name}   cos = {res['cos'][i]:.3f}", res["medoids"][:, i])
                       for name, res in results.items()]
            plot_sbs(panels, f"{sbs_dir}/{sig}.png")
    else:
        # no truth: pair each W-NMF signature with its best KL-NMF match (Hungarian on cosine)
        idx, cos = match_signatures(wn["medoids"], kl["medoids"])
        for i in range(k):
            panels = [(wn["names"][i], wn["medoids"][:, i]),
                      (f"{kl['names'][idx[i]]}   cos to {wn['names'][i]} = {cos[i]:.3f}",
                       kl["medoids"][:, idx[i]])]
            plot_sbs(panels, f"{sbs_dir}/{wn['names'][i]}.png")
    print(f"Results saved to {path_to_save_output}")

    return results

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    run(args)

if __name__ == '__main__':
    main()
