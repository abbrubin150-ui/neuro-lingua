"""Independent numpy/scipy reference for scripts/audit/math_harness.ts (see docs/audits/MATH_AUDIT.md).

Usage: python3 scripts/audit/math_reference.py <harness-output.json>   (requires numpy, scipy)
"""
import json, math, sys
import numpy as np
from scipy import optimize, stats, special
from scipy.spatial import cKDTree

d = json.load(open(sys.argv[1]))
rows = []
def check(name, ours, ref, tol):
    ours = np.asarray(ours, dtype=float); ref = np.asarray(ref, dtype=float)
    err = float(np.max(np.abs(ours - ref))) if ours.size else 0.0
    rows.append((name, 'OK' if err <= tol else 'MISMATCH', err, tol))

# --- logistic regression MLE (intercept + 2 features)
X = np.array(d['propensity']['X']); Z = np.array(d['propensity']['Z'], dtype=float)
Xd = np.column_stack([np.ones(len(X)), X])
nll = lambda b: np.sum(np.logaddexp(0, Xd @ b) - Z * (Xd @ b))
grad = lambda b: Xd.T @ (special.expit(Xd @ b) - Z)
b = optimize.minimize(nll, np.zeros(3), jac=grad, method='BFGS', options={'gtol': 1e-10}).x
coef = np.array(d['propensity']['result']['coefficients'])
check('propensity coefficients (logistic MLE)', coef, b, 1e-3)
check('propensity scores', d['propensities'], special.expit(Xd @ b), 1e-3)

# --- OLS outcome model: Y ~ 1 + Z + X
Y = np.array(d['outcomeModel']['Y']); om = d['outcomeModel']['om']
D = np.column_stack([np.ones(len(X)), Z, X])
beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
check('outcome model intercept', om['intercept'], beta[0], 1e-6)
check('outcome model treatment effect', om['treatmentEffect'], beta[1], 1e-6)
check('outcome model feature coefficients', om['featureCoefficients'], beta[2:], 1e-6)

# --- AIPW
pi = np.array(d['propensities']); muA = np.array(d['aipw']['muA']); muB = np.array(d['aipw']['muB'])
psi = Z * (Y - muB) / pi + muB - ((1 - Z) * (Y - muA) / (1 - pi) + muA)
check('AIPW estimate', d['aipw']['result']['estimate'], psi.mean(), 1e-9)
check('AIPW standard error', d['aipw']['result']['standardError'], psi.std(ddof=1) / math.sqrt(len(psi)), 1e-9)

# --- IPW (Horvitz-Thompson) for comparison
ht = Z * Y / pi - (1 - Z) * Y / (1 - pi)
w1 = Z / pi; w0 = (1 - Z) / (1 - pi)
hajek = (w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum()
rows.append(('IPW estimate (TS) vs Horvitz-Thompson', 'INFO', abs(d['ipw']['estimate'] - ht.mean()), 0))
rows.append(('IPW estimate (TS) vs Hajek', 'INFO', abs(d['ipw']['estimate'] - hajek), 0))

# --- power / MDE (two-sided z test)
za = stats.norm.ppf(0.975)
pw = [1 - stats.norm.cdf(za - e / 0.12) + stats.norm.cdf(-za - e / 0.12) for e in [0.1, 0.25, 0.5]]
check('power', d['power'], pw, 1e-6)
check('minimum detectable effect', d['mde'], [(za + stats.norm.ppf(p)) * 0.12 for p in [0.5, 0.8, 0.9]], 1e-4)

# --- variance / covariance
x = np.array(d['variance']['x'])
check('sample variance', d['variance']['v'], x.var(ddof=1), 1e-9)
check('covariance', d['variance']['cov'], np.cov(x, Z[:50], ddof=1)[0, 1], 1e-9)

# --- KSG (Kraskov alg. 1) reference implementation on the same samples is not exported;
#     compare with the analytic MI of the bivariate Gaussian instead
rho = d['ksg']['rho']; true_mi = -0.5 * math.log(1 - rho ** 2)
rows.append(('KSG MI vs analytic (rho=0.8)', 'INFO', abs(d['ksg']['mi']['mutualInformation'] - true_mi), 0))
G = np.array(d['ksg']['G']); k = 3; N = len(G)
tree = cKDTree(G); eps = tree.query(G, k + 1, p=np.inf)[0][:, -1]
tx = cKDTree(G[:, :1]); ty = cKDTree(G[:, 1:])
nx = np.array([len(tx.query_ball_point(G[i, :1], eps[i] - 1e-15, p=np.inf)) - 1 for i in range(N)])
ny = np.array([len(ty.query_ball_point(G[i, 1:], eps[i] - 1e-15, p=np.inf)) - 1 for i in range(N)])
ksg = special.digamma(k) + special.digamma(N) - np.mean(special.digamma(nx + 1) + special.digamma(ny + 1))
check('KSG (Kraskov alg. 1) vs independent implementation', d['ksg']['mi']['mutualInformation'], ksg, 1e-7)

# --- Kozachenko-Leonenko entropy of N(0, sigma^2)
true_h = 0.5 * math.log(2 * math.pi * math.e * d['knnEntropy']['sigma'] ** 2)
check('kNN entropy vs analytic N(0,4)', d['knnEntropy']['h'], true_h, 0.05)

# --- compensated sums
vals = d['sums']['vals']; exact = math.fsum(vals)
rows.append(('kahanSum on [1e16, 1, -1e16, ...] (known Kahan limitation)', 'INFO', abs(d['sums']['kahan'] - exact), 0))
check('neumaierSum', d['sums']['neumaier']['sum'], exact, 1e-12)
rows.append(('pairwiseSum on [1e16, 1, -1e16, ...] (not compensated)', 'INFO', abs(d['sums']['pairwise'] - exact), 0))
check('kahanSum of 0.1 x 1e6', d['sums']['tenth'], math.fsum([0.1] * 1000000), 1e-9)
rows.append(('naive sum of 0.1 x 1e6 (reference error)', 'INFO', abs(d['sums']['tenthNaive'] - math.fsum([0.1] * 1000000)), 0))

# --- norms / condition number
M = np.array(d['norms']['M'])
check('Frobenius norm', d['norms']['fro'], np.linalg.norm(M, 'fro'), 1e-10)
check('spectral norm', d['norms']['spec']['norm'], np.linalg.norm(M, 2), 1e-4)
check('condition number kappa_1', d['norms']['cond'] / np.linalg.cond(M[:5], 1), 1.0, 1e-9)

# --- stable variance of 1e9 + noise
sv = np.array(d['stableVariance']['sv'])
check('stableVariance.variance', d['stableVariance']['r']['variance'], sv.var(ddof=1), 1e-6)

# --- graph Laplacian spectrum
A = np.array(d['graph']['A']); W = (A + A.T) / 2
deg = W.sum(1); L = np.diag(deg) - W
Ln = np.eye(len(W)) - np.diag(deg ** -0.5) @ W @ np.diag(deg ** -0.5)
check('Laplacian (unnormalized, symmetrized)', d['graph']['L'], L, 1e-9)
check('Laplacian (normalized, symmetrized)', d['graph']['Lnorm'], Ln, 1e-9)
ev = np.sort(np.linalg.eigvalsh(Ln))
check('Laplacian eigenvalues', np.sort(d['graph']['analysis']['eigenvalues']), ev, 1e-4)
check('algebraic connectivity', d['graph']['analysis']['algebraicConnectivity'], ev[1], 1e-4)

# --- randomized SVD
B = np.array(d['rsvd']['B']); s = np.linalg.svd(B, compute_uv=False)
check('randomizedSVD top-3 singular values', d['rsvd']['r']['singularValues'], s[:3], 1e-6)

width = max(len(r[0]) for r in rows)
for name, status, err, tol in rows:
    print(f"{name:<{width}}  {status:8}  err={err:.3g}" + (f"  tol={tol:g}" if status != 'INFO' else ''))
