"""Gjenskap beregningene og figurene i PCA-rapporten.

Kjør med: python verify_experiment.py
Resultatfiler og figurer lagres ved siden av skriptet.
Datasettet med håndskrevne sifre følger med scikit-learn.
"""
from pathlib import Path
import csv
import json
import platform
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SEED = 42
CANDIDATES = [2, 5, 10, 15, 20, 30, 40, 64]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def fit_pca(x):
    """Finn middelverdi, egenverdier og hovedretninger fra radene i x."""
    mean = x.mean(axis=0)
    xc = x - mean
    cov = xc.T @ xc / (len(x) - 1)
    # eigh brukes fordi kovariansmatrisen er symmetrisk.
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    # Små negative egenverdier kan oppstå ved avrunding i flyttallsregningen.
    return mean, np.maximum(values[order], 0), vectors[:, order]


def predict_centroid(xtrain, ytrain, xquery):
    """Gi hvert bilde etiketten til det nærmeste klassegjennomsnittet."""
    classes = np.unique(ytrain)
    centers = np.stack([xtrain[ytrain == c].mean(axis=0) for c in classes])
    squared_distances = ((xquery[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return classes[np.argmin(squared_distances, axis=1)]


def verify_math():
    """Kontroller firepunktseksemplet og tegn projeksjonen på første retning."""
    x = np.array([[5., 4.], [4., 5.], [2., 1.], [1., 2.]])
    mean, ev, u = fit_pca(x)
    reconstructed = (x - mean) @ u[:, :1] @ u[:, :1].T + mean
    expected = np.array([[4.5, 4.5], [4.5, 4.5], [1.5, 1.5], [1.5, 1.5]])
    assert np.allclose(mean, [3, 3])
    assert np.allclose(ev, [6, 2/3])
    assert np.allclose(reconstructed, expected)
    assert np.isclose(((x - reconstructed) ** 2).sum(), 2)
    assert np.isclose(ev[0] / ev.sum(), .9)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot([.5, 5.5], [.5, 5.5], color='#14616c', label='First principal direction')
    for original, projected in zip(x, reconstructed):
        ax.plot([original[0], projected[0]], [original[1], projected[1]], ':', color='#777777')
    ax.scatter(x[:, 0], x[:, 1], color='#222222', s=38, label='Original data', zorder=4)
    ax.scatter(expected[:, 0], expected[:, 1], facecolors='white', edgecolors='#14616c',
               marker='s', s=52, label='Reconstruction', zorder=5)
    ax.scatter(*mean, marker='x', color='#b54040', s=60, label='Mean')
    ax.set(xlabel='First coordinate', ylabel='Second coordinate', xlim=(.4, 5.6), ylim=(.4, 5.6))
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(ROOT / 'worked-example.pdf')
    plt.close(fig)
    return {'mean': mean.tolist(), 'eigenvalues': ev.tolist(),
            'explained_variance_ratio': .9, 'squared_error': 2.0}


def main():
    worked = verify_math()
    dataset = load_digits()
    x = dataset.data.astype(np.float64) / 16.0
    y = dataset.target
    indices = np.arange(len(x))
    # Testsettet holdes utenfor både tilpasningen og valget av antall komponenter.
    fit_indices, test_indices = train_test_split(indices, test_size=.2, random_state=SEED, stratify=y)
    train_indices, val_indices = train_test_split(fit_indices, test_size=.25,
                                                 random_state=SEED, stratify=y[fit_indices])
    assert not (set(train_indices) & set(val_indices) or set(train_indices) & set(test_indices)
                or set(val_indices) & set(test_indices))
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(x)
    mean, ev, vectors = fit_pca(x[train_indices])
    validation = []
    for k in CANDIDATES:
        u = vectors[:, :k]
        pred = predict_centroid((x[train_indices] - mean) @ u, y[train_indices],
                                (x[val_indices] - mean) @ u)
        correct = int((pred == y[val_indices]).sum())
        val_reconstruction = (x[val_indices] - mean) @ u @ u.T + mean
        validation.append({'k': k, 'correct': correct, 'n': len(val_indices),
                           'accuracy': correct / len(val_indices),
                           'train_variance_ratio': float(ev[:k].sum() / ev.sum()),
                           'validation_mse': float(np.mean((x[val_indices]-val_reconstruction)**2))})
    # Ved lik valideringsnøyaktighet velges det minste antallet komponenter.
    selected = min(validation, key=lambda r: (-r['correct'], r['k']))['k']
    # Tilpass på trening + validering etter at k er valgt. Testsettet brukes først nå.
    mean, ev, vectors = fit_pca(x[fit_indices])
    u = vectors[:, :selected]
    zfit = (x[fit_indices] - mean) @ u
    ztest = (x[test_indices] - mean) @ u
    prediction = predict_centroid(zfit, y[fit_indices], ztest)
    baseline = predict_centroid(x[fit_indices], y[fit_indices], x[test_indices])
    # Alle retningene sammen bevarer avstandene og skal gi samme svar som pikslene.
    full_rotation = predict_centroid((x[fit_indices] - mean) @ vectors, y[fit_indices],
                                     (x[test_indices] - mean) @ vectors)
    assert np.array_equal(baseline, full_rotation)
    reconstruction = ztest @ u.T + mean
    xc = x[fit_indices] - mean
    squared_error = float(np.sum((xc - xc @ u @ u.T) ** 2))
    eigenvalue_error = float((len(fit_indices) - 1) * ev[selected:].sum())
    # Sjekk identiteten mellom rekonstruksjonsfeil og forkastede egenverdier.
    assert np.isclose(squared_error, eigenvalue_error, rtol=1e-10, atol=1e-10)
    reference = PCA(n_components=selected, svd_solver='full').fit(x[fit_indices])
    # Fortegn på egenvektorer kan variere; projeksjonsmatrisen skal likevel bli lik.
    projector_difference = float(np.max(np.abs(u @ u.T - reference.components_.T @ reference.components_)))
    assert projector_difference < 1e-10
    assert np.allclose(ev[:selected], reference.explained_variance_, rtol=1e-10, atol=1e-12)
    # Med 32 bilder og 64 piksler kan den mindre Gram-matrisen sammenlignes med S.
    small = x[train_indices[:32]]
    _, _, small_u = fit_pca(small)
    a = (small - small.mean(axis=0)).T
    d, v = np.linalg.eigh(a.T @ a)
    positive = d > 1e-10
    dual_u = a @ v[:, positive] / np.sqrt(d[positive])[None, :]
    rank = int(positive.sum())
    dual_difference = float(np.max(np.abs(dual_u @ dual_u.T - small_u[:, :rank] @ small_u[:, :rank].T)))
    assert rank <= 31 and dual_difference < 1e-9
    correct = int((prediction == y[test_indices]).sum())
    baseline_correct = int((baseline == y[test_indices]).sum())
    output = {'seed': SEED, 'candidates': CANDIDATES, 'n': len(x), 'p': x.shape[1],
              'train_n': len(train_indices), 'validation_n': len(val_indices), 'test_n': len(test_indices),
              'development_n': len(fit_indices), 'selected_k': selected, 'validation': validation,
              'test_correct': correct, 'test_accuracy': correct / len(test_indices),
              'baseline_correct': baseline_correct, 'baseline_accuracy': baseline_correct / len(test_indices),
              'retained_variance': float(ev[:selected].sum() / ev.sum()),
              'test_mse': float(np.mean((x[test_indices] - reconstruction) ** 2)),
              'train_sse': squared_error, 'eigenvalue_sse': eigenvalue_error,
              'svd_projector_max_difference': projector_difference,
              'dual_projector_max_difference': dual_difference, 'dual_rank': rank,
              'worked_example': worked,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__}}
    (ROOT / 'verified-results.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    (ROOT / 'split-indices.json').write_text(json.dumps({'training': train_indices.tolist(),
               'validation': val_indices.tolist(), 'test': test_indices.tolist()}), encoding='utf-8')
    with (ROOT / 'test-predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset_index', 'true_label', 'pca_prediction', 'baseline_prediction'])
        writer.writerows(zip(test_indices, y[test_indices], prediction, baseline))
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(np.arange(1, 65), np.cumsum(ev) / ev.sum() * 100, color='#14616c', lw=2)
    ax.axvline(selected, color='#777777', linestyle=':', lw=1)
    retained = ev[:selected].sum() / ev.sum() * 100
    ax.scatter([selected], [retained], color='#14616c', zorder=5)
    ax.annotate(f'k = {selected}: {retained:.2f}%', (selected, retained), xytext=(-8, -25),
                textcoords='offset points', ha='right')
    ax.set(xlabel='Number of principal components, k', ylabel='Cumulative variance retained (%)',
           xlim=(1, 64), ylim=(0, 103))
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / 'retained-variance.pdf')
    plt.close(fig)
    # Bruk første testbilde med hver av disse etikettene, så figurvalget kan gjentas.
    selected_rows = [int(np.flatnonzero(y[test_indices] == label)[0]) for label in [0, 2, 5, 8]]
    fig, axes = plt.subplots(3, 4, figsize=(6.2, 4.7))
    for j, row in enumerate(selected_rows):
        for i, k in enumerate([64, 5, selected]):
            image = x[test_indices[row]] if i == 0 else (
                (x[test_indices[row]] - mean) @ vectors[:, :k] @ vectors[:, :k].T + mean)
            axes[i, j].imshow(image.reshape(8, 8), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f'Label {y[test_indices[row]]}')
            if j == 0:
                axes[i, j].set_ylabel('Original' if i == 0 else f'k = {k}')
    fig.tight_layout()
    fig.savefig(ROOT / 'digit-reconstructions.pdf')
    plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
