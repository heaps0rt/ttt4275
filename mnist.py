from __future__ import annotations

import argparse
import csv
import math
import struct
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM_CLASSES = 10


def read_idx_images(path: Path) -> np.ndarray:
	with path.open("rb") as f:
		magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
		if magic != 2051:
			raise ValueError(f"bad image magic in {path}: {magic}")
		return np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows * cols)


def read_idx_labels(path: Path) -> np.ndarray:
	with path.open("rb") as f:
		magic, count = struct.unpack(">II", f.read(8))
		if magic != 2049:
			raise ValueError(f"bad label magic in {path}: {magic}")
		labels = np.frombuffer(f.read(), dtype=np.uint8)
	if labels.size != count:
		raise ValueError(f"label count mismatch in {path}")
	return labels


def load_mnist(data_dir: Path):
	mat_path = data_dir / "data_all.mat"
	if mat_path.exists():
		data = loadmat(mat_path)
		return (
			data["trainv"].astype(np.float32),
			data["trainlab"].reshape(-1).astype(np.int64),
			data["testv"].astype(np.float32),
			data["testlab"].reshape(-1).astype(np.int64),
		)
	return (
		read_idx_images(data_dir / "train_images.bin").astype(np.float32),
		read_idx_labels(data_dir / "train_labels.bin").astype(np.int64),
		read_idx_images(data_dir / "test_images.bin").astype(np.float32),
		read_idx_labels(data_dir / "test_labels.bin").astype(np.int64),
	)


def confusion_matrix(y_true, y_pred, n_classes):
	cm = np.zeros((n_classes, n_classes), dtype=np.int64)
	np.add.at(cm, (y_true, y_pred), 1)
	return cm


def chunked_sqdist(test, templates, chunk_size):
	n = test.shape[0]
	tnorm = np.sum(templates * templates, axis=1)
	tT = templates.T
	for start in range(0, n, chunk_size):
		end = min(start + chunk_size, n)
		chunk = test[start:end]
		cnorm = np.sum(chunk * chunk, axis=1, keepdims=True)
		d = cnorm + tnorm[None, :] - 2.0 * (chunk @ tT)
		np.maximum(d, 0.0, out=d)
		yield start, end, d


def predict_nn(templates, template_labels, test, chunk_size):
	n = test.shape[0]
	preds = np.empty(n, dtype=np.int64)
	for start, end, d in chunked_sqdist(test, templates, chunk_size):
		preds[start:end] = template_labels[np.argmin(d, axis=1)]
	return preds


def knn_vote(labels, distances, n_classes):
	counts = np.bincount(labels, minlength=n_classes)
	top = counts.max()
	tied = np.flatnonzero(counts == top)
	if tied.size == 1:
		return int(tied[0])
	best_class, best_mean = int(tied[0]), np.inf
	for c in tied:
		m = float(distances[labels == c].mean())
		if m < best_mean:
			best_class, best_mean = int(c), m
	return best_class


def predict_knn(templates, template_labels, test, k, chunk_size, n_classes):
	n = test.shape[0]
	preds = np.empty(n, dtype=np.int64)
	k_eff = min(k, templates.shape[0])
	for start, end, d in chunked_sqdist(test, templates, chunk_size):
		k_idx = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
		k_labels = template_labels[k_idx]
		k_dists = np.take_along_axis(d, k_idx, axis=1)
		for i in range(k_labels.shape[0]):
			preds[start + i] = knn_vote(k_labels[i], k_dists[i], n_classes)
	return preds


def cluster_templates(trainv, trainlab, m_per_class, seed):
	vectors, labels = [], []
	for c in range(NUM_CLASSES):
		cls = trainv[trainlab == c]
		km = KMeans(n_clusters=m_per_class, random_state=seed, n_init="auto")
		km.fit_predict(cls)
		vectors.append(km.cluster_centers_.astype(np.float32))
		labels.append(np.full(m_per_class, c, dtype=np.int64))
	return np.vstack(vectors), np.concatenate(labels)


def print_cm(cm, title):
	print(f"\n{title}")
	header = " " * 10 + " ".join(f"pred:{i:>4d}" for i in range(cm.shape[1]))
	print(header)
	for i, row in enumerate(cm):
		print(f"true:{i:>3d} " + " ".join(f"{v:9d}" for v in row))


def write_cm_csv(path, cm):
	with path.open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["true\\pred"] + list(range(cm.shape[1])))
		for i, row in enumerate(cm):
			w.writerow([i] + row.tolist())


def plot_examples(vectors, true_labels, pred_labels, indices, path, title, max_images):
	if indices.size == 0:
		return
	selected = indices[:max_images]
	n = selected.size
	ncols = min(6, n)
	nrows = math.ceil(n / ncols)
	fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows))
	axes_arr = np.atleast_1d(axes).reshape(-1)
	for ax in axes_arr[n:]:
		ax.axis("off")
	for ax, idx in zip(axes_arr, selected):
		ax.imshow(vectors[idx].reshape(28, 28), cmap="gray")
		ax.set_title(f"i={idx} t={true_labels[idx]} p={pred_labels[idx]}", fontsize=8)
		ax.axis("off")
	fig.suptitle(title)
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	fig.savefig(path, dpi=150)
	plt.close(fig)


def plot_cluster_grid(templates, path, m_per_class, per_class=8):
	fig, axes = plt.subplots(
		NUM_CLASSES, per_class, figsize=(per_class * 1.2, NUM_CLASSES * 1.2)
	)
	for c in range(NUM_CLASSES):
		start = c * m_per_class
		cls = templates[start : start + per_class]
		for k in range(per_class):
			ax = axes[c, k]
			ax.imshow(cls[k].reshape(28, 28), cmap="gray")
			ax.set_xticks([])
			ax.set_yticks([])
			for spine in ax.spines.values():
				spine.set_visible(False)
			if k == 0:
				ax.set_ylabel(f"class {c}", rotation=90, labelpad=4, fontsize=11)
	fig.suptitle(f"First {per_class} of the M={m_per_class} k-means templates per class",
	             y=0.995, fontsize=12)
	fig.tight_layout(rect=(0, 0, 1, 0.975))
	fig.savefig(path, dpi=150)
	plt.close(fig)


def parse_args():
	p = argparse.ArgumentParser(description="MNIST NN/KNN experiments")
	p.add_argument("--data-dir", type=Path, default=Path("data/MNIST_files"))
	p.add_argument("--chunk-size", type=int, default=400)
	p.add_argument("--clusters-per-class", type=int, default=64)
	p.add_argument("--k", type=int, default=7)
	p.add_argument("--max-plots", type=int, default=12)
	p.add_argument("--random-state", type=int, default=42)
	p.add_argument("--max-test", type=int, default=None)
	p.add_argument("--figures-dir", type=Path, default=Path("figures/mnist"))
	p.add_argument("--results-dir", type=Path, default=Path("results/mnist"))
	return p.parse_args()


def main():
	args = parse_args()
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	args.results_dir.mkdir(parents=True, exist_ok=True)

	print("Loading MNIST...")
	trainv, trainlab, testv, testlab = load_mnist(args.data_dir)
	if args.max_test is not None:
		testv, testlab = testv[: args.max_test], testlab[: args.max_test]
	print(f"train {trainv.shape}, test {testv.shape}")

	# Task 1: full-template NN
	print("\n=== Task 1: full-template NN ===")
	t0 = time.perf_counter()
	full_pred = predict_nn(trainv, trainlab, testv, args.chunk_size)
	full_time = time.perf_counter() - t0
	full_cm = confusion_matrix(testlab, full_pred, NUM_CLASSES)
	full_err = float(np.mean(full_pred != testlab))
	print_cm(full_cm, "full-template NN confusion matrix")
	print(f"error rate: {100 * full_err:.2f}%   time: {full_time:.2f} s")

	mis_idx = np.where(full_pred != testlab)[0]
	cor_idx = np.where(full_pred == testlab)[0]
	plot_examples(testv, testlab, full_pred, mis_idx,
	              args.figures_dir / "misclassified_nn.png",
	              "Misclassified (full-template NN)", args.max_plots)
	plot_examples(testv, testlab, full_pred, cor_idx,
	              args.figures_dir / "correct_nn.png",
	              "Correctly classified (full-template NN)", args.max_plots)

	# Task 2: clustered templates + NN/KNN
	print("\n=== Task 2: clustered NN and KNN ===")
	t0 = time.perf_counter()
	templates, template_labels = cluster_templates(
		trainv, trainlab, args.clusters_per_class, args.random_state,
	)
	cluster_time = time.perf_counter() - t0
	print(f"{templates.shape[0]} templates ({args.clusters_per_class}/class) in {cluster_time:.2f} s")
	plot_cluster_grid(templates, args.figures_dir / "cluster_grid.png",
	                  args.clusters_per_class)

	t0 = time.perf_counter()
	cnn_pred = predict_nn(templates, template_labels, testv, args.chunk_size)
	cnn_time = time.perf_counter() - t0
	cnn_cm = confusion_matrix(testlab, cnn_pred, NUM_CLASSES)
	cnn_err = float(np.mean(cnn_pred != testlab))
	print_cm(cnn_cm, "clustered NN confusion matrix")
	print(f"error rate: {100 * cnn_err:.2f}%   time: {cnn_time:.2f} s")

	t0 = time.perf_counter()
	knn_pred = predict_knn(templates, template_labels, testv,
	                       args.k, args.chunk_size, NUM_CLASSES)
	knn_time = time.perf_counter() - t0
	knn_cm = confusion_matrix(testlab, knn_pred, NUM_CLASSES)
	knn_err = float(np.mean(knn_pred != testlab))
	print_cm(knn_cm, f"clustered KNN (K={args.k}) confusion matrix")
	print(f"error rate: {100 * knn_err:.2f}%   time: {knn_time:.2f} s")

	print("\n=== Summary ===")
	print(f"full NN    : err {100 * full_err:5.2f}%   time {full_time:7.2f} s")
	print(f"cluster NN : err {100 * cnn_err:5.2f}%   time {cnn_time:7.2f} s")
	print(f"cluster KNN: err {100 * knn_err:5.2f}%   time {knn_time:7.2f} s (K={args.k})")
	print(f"clustering : {cluster_time:.2f} s   NN speedup {full_time / cnn_time:.1f}x")

	with (args.results_dir / "summary.csv").open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["system", "error_%", "time_s"])
		w.writerow(["full_nn", f"{100 * full_err:.4f}", f"{full_time:.6f}"])
		w.writerow(["clustered_nn", f"{100 * cnn_err:.4f}", f"{cnn_time:.6f}"])
		w.writerow([f"clustered_knn_k{args.k}", f"{100 * knn_err:.4f}", f"{knn_time:.6f}"])
		w.writerow(["kmeans_preprocessing", "", f"{cluster_time:.6f}"])

	write_cm_csv(args.results_dir / "confusion_full_nn.csv", full_cm)
	write_cm_csv(args.results_dir / "confusion_cluster_nn.csv", cnn_cm)
	write_cm_csv(args.results_dir / f"confusion_cluster_knn_k{args.k}.csv", knn_cm)

	print(f"\nFigures -> {args.figures_dir}")
	print(f"Results -> {args.results_dir}")


if __name__ == "__main__":
	main()
