import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
FEATURE_NAMES = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


@dataclass
class Trained:
	W: np.ndarray
	alpha: float
	mse_history: list[float]
	error_history: list[float]
	epochs: int
	converged: bool


def load_iris(path: Path) -> tuple[np.ndarray, np.ndarray]:
	# Parse iris.data: 4 feature floats + class name per line; map names to 0/1/2.
	label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
	features, labels = [], []
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		parts = line.split(",")
		if len(parts) != 5:
			continue
		features.append([float(v) for v in parts[:4]])
		labels.append(label_to_index[parts[4]])
	return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def classwise_split(x, y, train_per_class, from_start):
	# Take the same number of samples from every class so the split is balanced.
	# from_start=True  -> first N per class for training (Split A).
	# from_start=False -> last  N per class for training (Split B).
	train_idx, test_idx = [], []
	for c in range(len(CLASS_NAMES)):
		idx = np.where(y == c)[0]
		if from_start:
			train_idx.extend(idx[:train_per_class].tolist())
			test_idx.extend(idx[train_per_class:].tolist())
		else:
			train_idx.extend(idx[-train_per_class:].tolist())
			test_idx.extend(idx[:-train_per_class].tolist())
	train_idx = np.asarray(train_idx, dtype=np.int64)
	test_idx = np.asarray(test_idx, dtype=np.int64)
	return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def standardize(x_train, x_test):
	# Center and scale features using only the training set's mean/std.
	# The std==0 guard avoids dividing by zero if a feature is constant.
	mean = x_train.mean(axis=0)
	std = x_train.std(axis=0)
	std[std == 0.0] = 1.0
	return (x_train - mean) / std, (x_test - mean) / std


def one_hot(y, n_classes):
	# Turn class labels into target rows: a 1 in the true-class column, 0 elsewhere.
	t = np.zeros((y.size, n_classes), dtype=np.float64)
	t[np.arange(y.size), y] = 1.0
	return t


def sigmoid(z):
	# Sigmoid written to avoid overflow on large positive or negative inputs.
	out = np.empty_like(z, dtype=np.float64)
	pos = z >= 0
	out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
	ez = np.exp(z[~pos])
	out[~pos] = ez / (1.0 + ez)
	return out


def add_bias(x):
	# Add a column of 1s so the bias term can be stored in the weight matrix.
	return np.hstack((x, np.ones((x.shape[0], 1), dtype=x.dtype)))


def predict(W, x):
	# Pick the class with the largest output for each sample.
	return np.argmax(add_bias(x) @ W, axis=1)


def train(x, y, alphas, max_epochs, tol):
	# Linear classifier with sigmoid output, trained by minimising MSE.
	# Tries each step size and keeps the one with the lowest final training error.
	n_classes = len(CLASS_NAMES)
	xb = add_bias(x)
	t = one_hot(y, n_classes)

	best: Trained | None = None
	for alpha in alphas:
		W = np.zeros((xb.shape[1], n_classes), dtype=np.float64)
		mse_hist: list[float] = []
		err_hist: list[float] = []
		prev_mse = np.inf
		converged = False

		for _ in range(max_epochs):
			z = xb @ W
			g = sigmoid(z)
			diff = g - t
			# Mean-square error summed over samples and classes.
			mse = 0.5 * float(np.sum(diff * diff))
			# If training diverged, drop this step size.
			if not np.isfinite(mse) or mse > 1e10:
				break
			err = float(np.mean(np.argmax(z, axis=1) != y))
			mse_hist.append(mse)
			err_hist.append(err)

			# Stop once the MSE barely changes; the warmup keeps us from stopping too early.
			if len(mse_hist) > 20 and abs(prev_mse - mse) < tol:
				converged = True
				break

			# Gradient step. The g*(1-g) factor is the derivative of the sigmoid.
			W -= alpha * (xb.T @ (diff * g * (1.0 - g)))
			prev_mse = mse

		result = Trained(
			W=W, alpha=alpha,
			mse_history=mse_hist, error_history=err_hist,
			epochs=len(mse_hist), converged=converged,
		)
		if best is None or result.error_history[-1] < best.error_history[-1]:
			best = result
	assert best is not None
	return best


def confusion_matrix(y_true, y_pred, n_classes):
	# cm[i, j] = number of samples with true class i predicted as class j.
	cm = np.zeros((n_classes, n_classes), dtype=np.int64)
	np.add.at(cm, (y_true, y_pred), 1)
	return cm


def evaluate(x_train, y_train, x_test, y_test, alphas, max_epochs, tol):
	# Standardise, train, and report errors and confusion matrices on both sets.
	x_train_s, x_test_s = standardize(x_train, x_test)
	trained = train(x_train_s, y_train, alphas, max_epochs, tol)
	train_pred = predict(trained.W, x_train_s)
	test_pred = predict(trained.W, x_test_s)
	return {
		"trained": trained,
		"train_cm": confusion_matrix(y_train, train_pred, len(CLASS_NAMES)),
		"test_cm": confusion_matrix(y_test, test_pred, len(CLASS_NAMES)),
		"train_error": float(np.mean(train_pred != y_train)),
		"test_error": float(np.mean(test_pred != y_test)),
	}


def feature_overlap(x, y, bins):
	# Score how much the class histograms overlap for each feature.
	# Lower score = the classes are more separated on that feature.
	scores = np.zeros(x.shape[1])
	for j in range(x.shape[1]):
		# Use the same bin edges for every class so the histograms can be compared.
		edges = np.linspace(x[:, j].min(), x[:, j].max(), bins + 1)
		hists = []
		for c in range(len(CLASS_NAMES)):
			h, _ = np.histogram(x[y == c, j], bins=edges)
			h = h.astype(float)
			h /= max(h.sum(), 1.0)
			hists.append(h)
		total, pairs = 0.0, 0
		for i in range(len(hists)):
			for k in range(i + 1, len(hists)):
				total += float(np.minimum(hists[i], hists[k]).sum())
				pairs += 1
		scores[j] = total / pairs
	return scores


def feature_subset(scores, n_features):
	# Pick the n features with the smallest overlap (kept in their original order).
	return sorted(np.argsort(scores)[:n_features].tolist())


def feature_names(indices):
	return ", ".join(FEATURE_NAMES[i] for i in indices)


def plot_histograms(x, y, path, bins):
	# Per-feature histogram, one curve per class, to show feature separability.
	fig, axes = plt.subplots(2, 2, figsize=(10, 7))
	for ax, j in zip(axes.flat, range(4)):
		for c, name in enumerate(CLASS_NAMES):
			ax.hist(
				x[y == c, j], bins=bins, alpha=0.55, density=True,
				color=CLASS_COLORS[c], label=name,
			)
		ax.set_title(FEATURE_NAMES[j])
		ax.set_xlabel("value [cm]")
		ax.set_ylabel("density")
		ax.grid(alpha=0.3)
	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
	           bbox_to_anchor=(0.5, -0.01))
	fig.tight_layout(rect=(0, 0.04, 1, 1))
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def plot_training_curves(trained: Trained, path: Path, split_label: str) -> None:
	# MSE and training error rate over the epochs of the best run.
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
	ax1.plot(trained.mse_history, color="#1f6db0", linewidth=1.4)
	ax1.set_xlabel("epoch")
	ax1.set_ylabel("MSE")
	ax1.set_title(f"MSE vs. epoch (alpha={trained.alpha})")
	ax1.grid(alpha=0.3)
	ax2.plot(np.array(trained.error_history) * 100, color="#c44e4e", linewidth=1.4)
	ax2.set_xlabel("epoch")
	ax2.set_ylabel("training error rate [%]")
	ax2.set_title("Training error vs. epoch")
	ax2.grid(alpha=0.3)
	fig.suptitle(split_label, fontsize=11)
	fig.tight_layout(rect=(0, 0, 1, 0.94))
	fig.savefig(path, dpi=150)
	plt.close(fig)


def plot_decision_regions(x_train, y_train, feat_idx, alphas, max_epochs, tol, path):
	# Retrain on just two features and shade the plane by predicted class.
	x_sub = x_train[:, feat_idx]
	x_sub_s = (x_sub - x_sub.mean(0)) / x_sub.std(0)
	trained = train(x_sub_s, y_train, alphas, max_epochs, tol)

	# Grid covering the data with a small margin around it.
	pad = 0.5
	g0 = np.linspace(x_sub_s[:, 0].min() - pad, x_sub_s[:, 0].max() + pad, 400)
	g1 = np.linspace(x_sub_s[:, 1].min() - pad, x_sub_s[:, 1].max() + pad, 400)
	xx, yy = np.meshgrid(g0, g1)
	grid = np.column_stack([xx.ravel(), yy.ravel()])
	preds = predict(trained.W, grid).reshape(xx.shape)

	fig, ax = plt.subplots(figsize=(7, 5.5))
	bg = ListedColormap(["#d6e4f2", "#fbe5cd", "#d6ecd4"])
	ax.pcolormesh(xx, yy, preds, cmap=bg, shading="auto")
	for c, name in enumerate(CLASS_NAMES):
		mask = y_train == c
		ax.scatter(
			x_sub_s[mask, 0], x_sub_s[mask, 1],
			s=32, edgecolor="black", linewidth=0.4,
			c=[CLASS_COLORS[c]], label=name, zorder=2,
		)
	ax.set_xlabel(f"{FEATURE_NAMES[feat_idx[0]]} (standardised)")
	ax.set_ylabel(f"{FEATURE_NAMES[feat_idx[1]]} (standardised)")
	ax.set_title("Decision regions of the linear MSE classifier (2D)")
	ax.legend(loc="lower right", framealpha=0.9)
	fig.tight_layout()
	fig.savefig(path, dpi=150)
	plt.close(fig)


def print_confusion_matrix(cm):
	header = " " * 16 + " ".join(f"pred:{n[:6]:>6}" for n in CLASS_NAMES)
	print(header)
	for i, row in enumerate(cm):
		row_str = " ".join(f"{v:12d}" for v in row)
		print(f"true:{CLASS_NAMES[i][:10]:>10} {row_str}")


def run_part_one(x, y, alphas, max_epochs, tol, figures_dir: Path):
	# Train and evaluate the classifier on both Split A and Split B.
	print("\n=== Part 1: two train/test splits ===")
	results = []
	splits = [
		("split_a_first30_train_last20_test",
		 "Split A: first 30 train, last 20 test",
		 classwise_split(x, y, 30, True)),
		("split_b_last30_train_first20_test",
		 "Split B: last 30 train, first 20 test",
		 classwise_split(x, y, 30, False)),
	]
	for name, pretty, split in splits:
		x_tr, y_tr, x_te, y_te = split
		ex = evaluate(x_tr, y_tr, x_te, y_te, alphas, max_epochs, tol)
		trained: Trained = ex["trained"]
		print(f"\n--- {pretty} ---")
		print(f"alpha={trained.alpha}, epochs={trained.epochs}, converged={trained.converged}")
		print(f"train error: {100 * ex['train_error']:.2f}%")
		print_confusion_matrix(ex["train_cm"])
		print(f"test error: {100 * ex['test_error']:.2f}%")
		print_confusion_matrix(ex["test_cm"])
		plot_training_curves(trained, figures_dir / f"training_curves_{name}.png", pretty)
		results.append({"name": name, **ex})
	return results


def run_part_two(x, y, alphas, max_epochs, tol, bins, figures_dir: Path):
	# Drop features one at a time (worst overlap first) and see how it affects accuracy.
	print("\n=== Part 2: feature reduction ===")
	x_tr, y_tr, x_te, y_te = classwise_split(x, y, 30, True)
	plot_histograms(x_tr, y_tr, figures_dir / "feature_histograms.png", bins=bins)

	overlap = feature_overlap(x_tr, y_tr, bins=bins)
	print("Histogram-overlap per feature (smaller = better separation):")
	for j, s in enumerate(overlap):
		print(f"  {FEATURE_NAMES[j]:<14}= {s:.4f}")

	results = []
	for n in [4, 3, 2, 1]:
		idx = feature_subset(overlap, n)
		ex = evaluate(x_tr[:, idx], y_tr, x_te[:, idx], y_te, alphas, max_epochs, tol)
		print(f"\n--- {n} feature(s): {feature_names(idx)} ---")
		print(f"train error: {100 * ex['train_error']:.2f}%")
		print_confusion_matrix(ex["train_cm"])
		print(f"test error: {100 * ex['test_error']:.2f}%")
		print_confusion_matrix(ex["test_cm"])
		results.append({"n": n, "indices": idx, "names": feature_names(idx), **ex})

	two_best = feature_subset(overlap, 2)
	plot_decision_regions(
		x_tr, y_tr, two_best, alphas, max_epochs, tol,
		figures_dir / "decision_regions.png",
	)
	return results, overlap


def write_csvs(results_dir: Path, part1, part2, overlap):
	# Save the numerical results from both parts as CSV files.
	results_dir.mkdir(parents=True, exist_ok=True)
	with (results_dir / "part1_summary.csv").open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["split", "alpha", "epochs", "converged", "train_err_%", "test_err_%"])
		for r in part1:
			t: Trained = r["trained"]
			w.writerow([
				r["name"], f"{t.alpha}", t.epochs, t.converged,
				f"{100 * r['train_error']:.4f}",
				f"{100 * r['test_error']:.4f}",
			])
	with (results_dir / "part2_summary.csv").open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["feature_count", "features", "train_err_%", "test_err_%"])
		for r in part2:
			w.writerow([
				r["n"], r["names"],
				f"{100 * r['train_error']:.4f}",
				f"{100 * r['test_error']:.4f}",
			])
	with (results_dir / "feature_overlap.csv").open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["feature", "overlap"])
		for name, s in zip(FEATURE_NAMES, overlap):
			w.writerow([name, f"{float(s):.6f}"])


def parse_args():
	p = argparse.ArgumentParser(description="Iris linear MSE classification")
	p.add_argument("--data", type=Path, default=Path("data/Iris_files/iris.data"))
	p.add_argument("--max-epochs", type=int, default=5000)
	p.add_argument("--tol", type=float, default=1e-8)
	p.add_argument("--bins", type=int, default=12)
	p.add_argument("--figures-dir", type=Path, default=Path("figures/iris"))
	p.add_argument("--results-dir", type=Path, default=Path("results/iris"))
	return p.parse_args()


def main():
	args = parse_args()
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	args.results_dir.mkdir(parents=True, exist_ok=True)

	alphas = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
	x, y = load_iris(args.data)

	part1 = run_part_one(x, y, alphas, args.max_epochs, args.tol, args.figures_dir)
	part2, overlap = run_part_two(
		x, y, alphas, args.max_epochs, args.tol, args.bins, args.figures_dir,
	)
	write_csvs(args.results_dir, part1, part2, overlap)
	print(f"\nFigures -> {args.figures_dir}")
	print(f"Results -> {args.results_dir}")


if __name__ == "__main__":
	main()
