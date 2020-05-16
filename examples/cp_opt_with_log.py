import numpy as np
import matplotlib.pyplot as plt

from tenkit import decomposition
import tenkit.utils


# Generate data tensor
rank = 5
true_tensor = decomposition.KruskalTensor.random_init((30, 40, 50), 5)
X = true_tensor.construct_tensor()
X_noisy = tenkit.utils.add_noise(X, noise_level=5)

# Generate decomposer and loggers
loggers = {
    "loss": decomposition.logging.LossLogger(),
    "SSE": decomposition.logging.SSELogger(),
    "reg": decomposition.logging.RegularisationPenaltyLogger(),
}
cp_opt = decomposition.CP_OPT(
    rank,
    1000,
    loggers=list(loggers.values()),
    convergence_tol=1e-10,
    print_frequency = 1,
    factor_penalties=[{'ridge':1e-5,}]*3
)

cp_opt.decomposition = true_tensor
cp_opt.fit(X_noisy)
print(cp_opt.explained_variance)
print(cp_opt.decomposition.factor_match_score(true_tensor))

# Plot logger output
fig, axes = plt.subplots(1, 3)

for i, (name, logger) in enumerate(loggers.items()):
    axes[i].set_title(name)
    axes[i].plot(logger.log_iterations, logger.log_metrics)
plt.show()