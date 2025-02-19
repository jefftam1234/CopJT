import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


### 📌 Step 1: Define the Gaussian Copula ###
class GaussianCopula:
    def __init__(self, rho: float):
        self.rho = rho  # Correlation coefficient

    def sample(self, U: np.ndarray) -> np.ndarray:
        # Convert independent uniforms into a bivariate normal sample with correlation rho
        Z = stats.norm.ppf(U)
        cov_matrix = [[1, self.rho], [self.rho, 1]]
        L = np.linalg.cholesky(cov_matrix)
        Z_correlated = np.dot(Z, L.T)
        return stats.norm.cdf(Z_correlated)


### 📌 Step 2: Define the Deterministic (Piecewise) Hazard Rate Model ###
class PiecewiseHazardRate:
    def __init__(self, pillar_times, hazard_rates):
        """
        pillar_times: list of time grid points (must start at 0)
        hazard_rates: list of hazard rates for each interval (length = len(pillar_times)-1)
        """
        self.pillar_times = np.array(pillar_times)
        self.hazard_rates = np.array(hazard_rates)

        if len(self.hazard_rates) != len(self.pillar_times) - 1:
            raise ValueError("Number of hazard rates must be one less than number of pillar times.")

        # Compute time differences for each interval.
        time_diffs = np.diff(self.pillar_times)

        # Build the cumulative hazard grid, starting at 0.
        cum_hazard = np.concatenate(([0], np.cumsum(time_diffs * self.hazard_rates)))
        self.cum_hazard = cum_hazard

        # Create the inverse mapping: given a cumulative hazard value h, find the corresponding time.
        # With a constant hazard rate (lambda), we have H(t)=lambda*t so that t = H(t)/lambda.
        self.inv_cdf = interp1d(self.cum_hazard, self.pillar_times, fill_value="extrapolate")

    def sample(self, U):
        """
        Transform a uniform U ~ Uniform(0,1) into a default time using:
            H = -ln(1-U)
            T = inv_cdf(H)
        In the constant hazard case (with pillars [0, T_max] and hazard_rate = lambda),
        this yields T = (-ln(1-U))/lambda.
        """
        cum_val = -np.log(1 - U)
        T_sample = self.inv_cdf(cum_val)
        return T_sample


### 📌 Step 3: Define the Copula Simulation Class ###
class CopulaSimulation:
    def __init__(self, copula, hazard_model, S0=100, sigma=0.2, num_samples=100000):
        self.copula = copula
        self.hazard_model = hazard_model
        self.S0 = S0
        self.sigma = sigma
        self.num_samples = num_samples

    def generate_joint_distribution(self):
        # Step 1: Generate independent uniform samples.
        U = np.random.uniform(size=(self.num_samples, 2))

        # Step 2: Apply the Gaussian copula to introduce dependence.
        V = self.copula.sample(U)

        # Step 3: Transform the first uniform sample using the hazard model.
        T = self.hazard_model.sample(V[:, 0])

        # Transform the second uniform sample to get a lognormal stock price.
        S = np.exp(np.log(self.S0) + self.sigma * stats.norm.ppf(V[:, 1]))

        return T, S


### 📌 Step 4: Define Visualization Functions ###
def plot_heatmap(T, S, copula_name):
    num_bins = 100
    T_bins = np.linspace(np.percentile(T, 1), np.percentile(T, 99), num_bins)
    S_bins = np.linspace(np.percentile(S, 1), np.percentile(S, 99), num_bins)
    heatmap, _, _ = np.histogram2d(T, S, bins=[T_bins, S_bins], density=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, extent=[T_bins[0], T_bins[-1], S_bins[0], S_bins[-1]],
               origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(label="Joint Density")
    plt.xlabel("Default Time T")
    plt.ylabel("Stock Price S_t")
    plt.title(f"Heatmap of Joint Distribution ({copula_name})")
    plt.show()


def plot_joint_cdf(T, S, copula_name):
    num_bins = 100
    T_bins = np.linspace(np.min(T), np.max(T), num_bins)
    S_bins = np.linspace(np.min(S), np.max(S), num_bins)
    joint_cdf = np.zeros((num_bins, num_bins))

    for i, t_val in enumerate(T_bins):
        for j, s_val in enumerate(S_bins):
            joint_cdf[i, j] = np.mean((T <= t_val) & (S <= s_val))

    plt.figure(figsize=(8, 6))
    plt.imshow(joint_cdf.T, extent=[T_bins[0], T_bins[-1], S_bins[0], S_bins[-1]],
               origin='lower', cmap='plasma', aspect='auto')
    plt.colorbar(label="Joint CDF Value")
    plt.xlabel("Default Time T")
    plt.ylabel("Stock Price S_t")
    plt.title(f"Empirical Joint CDF ({copula_name})")
    plt.show()


### 📌 Step 5: Run Simulation with Constant Deterministic Hazard Rate ###
# For a constant hazard rate lambda, we want H(t)=lambda*t.
# To mimic the exponential formula T = -ln(1-U)/lambda exactly, we use only two pillars: 0 and a high T_max.
lambda_val = 0.5
T_max = 100  # Set T_max high enough that almost all samples fall below it.
pillar_times = [0, T_max]
hazard_rates = [lambda_val]  # Constant hazard

# Build the hazard model.
hazard_model = PiecewiseHazardRate(pillar_times, hazard_rates)

# Use a Gaussian Copula with rho=0.3.
copula = GaussianCopula(rho=0.3)

# Create the simulation.
simulation = CopulaSimulation(copula=copula, hazard_model=hazard_model, S0=100, sigma=0.5, num_samples=100000)
T, S = simulation.generate_joint_distribution()

# Visualize the joint distribution.
plot_heatmap(T, S, "Gaussian Copula with Constant Deterministic Hazard")
plot_joint_cdf(T, S, "Gaussian Copula with Constant Deterministic Hazard")

# Using the deterministic hazard (piecewise with constant rate) simulation:
p1_determ = np.mean((T <= 2) & (S <= 110))  # ~0.41
p2_determ = np.mean((T <= 4) & (S <= 150))  # ~0.70

print(f"Probability of default before time 2 and stock price below 110 (Constant Hazard): {p1_determ:.2f}")
print(f"Probability of default before time 4 and stock price below 150 (Constant Hazard): {p2_determ:.2f}")