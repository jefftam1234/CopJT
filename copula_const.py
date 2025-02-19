import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import rankdata
from scipy.interpolate import griddata

### 📌 Step 1: Define the Base Copula Class ###
class BaseCopula:
    """Base class for all copulas."""
    def sample(self, U: np.ndarray) -> np.ndarray:
        """Transform independent uniform samples to dependent uniform samples."""
        raise NotImplementedError("Subclasses must implement this method.")

### 📌 Step 2: Implement Specific Copula Classes ###

# 1️⃣ Gaussian Copula
class GaussianCopula(BaseCopula):
    def __init__(self, rho: float):
        self.rho = rho  # Correlation coefficient

    def sample(self, U: np.ndarray) -> np.ndarray:
        """Apply Gaussian copula transformation."""
        mean = [0, 0]
        cov_matrix = [[1, self.rho], [self.rho, 1]]

        # Convert uniform to normal
        Z = stats.norm.ppf(U)

        # Apply correlation
        L = np.linalg.cholesky(cov_matrix)
        Z_correlated = np.dot(Z, L.T)

        # Convert back to uniform
        return stats.norm.cdf(Z_correlated)

# 2️⃣ Student-t Copula
class StudentTCopula(BaseCopula):
    def __init__(self, rho: float, df: int):
        self.rho = rho
        self.df = df  # Degrees of freedom

    def sample(self, U: np.ndarray) -> np.ndarray:
        """Apply Student-t copula transformation."""
        mean = [0, 0]
        cov_matrix = [[1, self.rho], [self.rho, 1]]

        # Convert uniform to standard normal
        Z = stats.norm.ppf(U)

        # Apply correlation
        L = np.linalg.cholesky(cov_matrix)
        Z_t = np.dot(Z, L.T)

        # Convert to t-distributed
        T_samples = stats.t.ppf(np.clip(stats.norm.cdf(Z_t), 1e-6, 1-1e-6), df=self.df)

        # Convert back to uniform
        return stats.t.cdf(T_samples, df=self.df)

# 3️⃣ Clayton Copula
class ClaytonCopula(BaseCopula):
    def __init__(self, theta: float):
        self.theta = theta

    def sample(self, U: np.ndarray) -> np.ndarray:
        """Apply Clayton copula transformation."""
        U1, U2 = U[:, 0], U[:, 1]
        U2_prime = (U1**(-self.theta) + U2**(-self.theta / (1 + self.theta)) - 1) ** (-1 / self.theta)
        return np.column_stack((U1, U2_prime))

### 📌 Step 3: Define the Copula Simulation Class ###
class CopulaSimulation:
    def __init__(self, copula: BaseCopula, lambda_T=0.1, S0=100, r=0.05, sigma=0.2, t=1.0, num_samples=100000):
        self.copula = copula
        self.lambda_T = lambda_T
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.t = t
        self.num_samples = num_samples

    def generate_joint_distribution(self):
        """Generate the joint distribution using a copula."""
        # Step 1: Generate Independent Uniform Samples
        U = np.random.uniform(size=(self.num_samples, 2))

        # Step 2: Apply the Copula Transformation
        V = self.copula.sample(U)

        # Step 3: Transform Back to Original Marginals
        X = -np.log(1 - V[:, 0]) / self.lambda_T  # Default time (Exponential)
        Y = np.exp(np.log(self.S0) + self.sigma * stats.norm.ppf(V[:, 1]))  # Stock price (Lognormal)

        return X, Y

### 📌 Step 4: Define the Visualization Functions ###

def plot_heatmap(T, S, copula_name):
    """Plot heatmap and 3D surface of joint distribution."""
    num_bins = 100
    T_bins = np.linspace(np.percentile(T, 1), np.percentile(T, 99), num_bins)
    S_bins = np.linspace(np.percentile(S, 1), np.percentile(S, 99), num_bins)

    heatmap, xedges, yedges = np.histogram2d(T, S, bins=[T_bins, S_bins], density=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, extent=[T_bins[0], T_bins[-1], S_bins[0], S_bins[-1]],
               origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(label="Joint Density")
    plt.xlabel("Default Time T")
    plt.ylabel("Stock Price S_t")
    plt.title(f"Heatmap of Joint Distribution ({copula_name})")
    plt.show()

    # 3D Surface Plot
    X, Y = np.meshgrid(T_bins[:-1], S_bins[:-1])
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, heatmap.T, cmap='viridis')

    ax.set_xlabel("Default Time T")
    ax.set_ylabel("Stock Price S_t")
    ax.set_zlabel("Joint Density")
    ax.set_title(f"3D Surface Plot of Joint Distribution ({copula_name})")
    plt.show()

def compute_joint_cdf(T, S):
    """Compute empirical joint CDF P(T ≤ t, S ≤ s)."""
    num_bins = 100
    T_bins = np.linspace(np.min(T), np.max(T), num_bins)
    S_bins = np.linspace(np.min(S), np.max(S), num_bins)

    joint_cdf = np.zeros((num_bins, num_bins))

    for i, t_val in enumerate(T_bins):
        for j, s_val in enumerate(S_bins):
            joint_cdf[i, j] = np.mean((T <= t_val) & (S <= s_val))  # Proportion of samples satisfying (T ≤ t, S ≤ s)

    return T_bins, S_bins, joint_cdf

def plot_joint_cdf(T, S, copula_name):
    """Plot empirical joint CDF."""
    T_bins, S_bins, joint_cdf = compute_joint_cdf(T, S)

    plt.figure(figsize=(8, 6))
    plt.imshow(joint_cdf.T, extent=[T_bins[0], T_bins[-1], S_bins[0], S_bins[-1]],
               origin='lower', cmap='plasma', aspect='auto')
    plt.colorbar(label="Joint CDF Value")
    plt.xlabel("Default Time T")
    plt.ylabel("Stock Price S_t")
    plt.title(f"Empirical Joint CDF ({copula_name})")
    plt.show()


### 📌 Step 5: Run Simulations for Different Copulas ###
copulas = {
    "Student-t": StudentTCopula(rho=0.3, df=5),
    "Clayton": ClaytonCopula(theta=2),
    "Gaussian": GaussianCopula(rho=0.3),
}

for name, copula in copulas.items():
    simulation = CopulaSimulation(copula=copula, lambda_T=0.5, S0=100, r=0.05, sigma=0.5, t=1.0, num_samples=100000)
    T, S = simulation.generate_joint_distribution()
    plot_heatmap(T, S, name)
    plot_joint_cdf(T, S, name)


# Using the deterministic hazard (piecewise with constant rate) simulation (gaussian for last)
p1_const = np.mean((T <= 2) & (S <= 110))  # ~0.41
p2_const = np.mean((T <= 4) & (S <= 150))  # ~0.70

print(f"Probability of default before time 2 and stock price below 110 (Constant Hazard): {p1_const:.2f}")
print(f"Probability of default before time 4 and stock price below 150 (Constant Hazard): {p2_const:.2f}")