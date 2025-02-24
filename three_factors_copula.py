import numpy as np
import scipy.stats as stats


#########################################
# Step 1: Define the Base Copula Class  #
#########################################
class BaseCopula:
    """Base class for all copulas."""

    def sample(self, U: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method.")


#########################################
# Step 2: Implement the Gaussian Copula #
#########################################
class GaussianCopula(BaseCopula):
    def __init__(self, corr_matrix: np.ndarray):
        """
        Parameters:
         - corr_matrix: a positive–definite correlation matrix of shape (d,d) for d factors.
        """
        self.corr_matrix = corr_matrix
        self.dim = corr_matrix.shape[0]
        self.L = np.linalg.cholesky(corr_matrix)

    def sample(self, U: np.ndarray) -> np.ndarray:
        # Convert independent uniforms to standard normals.
        Z = stats.norm.ppf(U)
        # Impose correlation.
        Z_corr = np.dot(Z, self.L.T)
        # Convert back to uniforms.
        return stats.norm.cdf(Z_corr)


####################################################
# Step 3: Define the Multi-Factor Simulation Class  #
####################################################
class MultiFactorSimulation:
    def __init__(self, copula: BaseCopula, sigma_fx: float, r0: float, sigma_ir: float, lambda_T: float,
                 num_samples: int = 100000):
        """
        Parameters:
         - sigma_fx: Volatility for FX; with FX = exp(sigma_fx * N(0,1)) and baseline 1,
                     the 5th percentile is exp(sigma_fx * Φ⁻¹(0.05)).
         - r0: Baseline interest rate (e.g. 0.01).
         - sigma_ir: Volatility for IR; with IR = r0 + sigma_ir * N(0,1), the 95th percentile is r0 + sigma_ir * 1.64485.
         - lambda_T: Hazard rate for default (T = -ln(1-U)/lambda_T).
         - num_samples: Number of samples.
        """
        self.copula = copula
        self.sigma_fx = sigma_fx
        self.r0 = r0
        self.sigma_ir = sigma_ir
        self.lambda_T = lambda_T
        self.num_samples = num_samples

    def generate_joint_distribution(self):
        """
        Generates samples for three factors:
         - FX: Lognormal with baseline 1.
         - IR: Normal with baseline r0.
         - T: Default time from an exponential with hazard rate lambda_T.
        """
        U = np.random.uniform(size=(self.num_samples, 3))
        V = self.copula.sample(U)

        # FX factor: lognormal transformation (baseline 1).
        FX = np.exp(self.sigma_fx * stats.norm.ppf(V[:, 0]))

        # IR factor: r0 plus normal noise.
        IR = self.r0 + self.sigma_ir * stats.norm.ppf(V[:, 1])

        # Default time: exponential default time.
        T = -np.log(1 - V[:, 2]) / self.lambda_T

        return FX, IR, T


#########################################
# Step 4: Set Up Parameters and Scenarios
#########################################
num_samples = 100000

# FX calibration: solve exp(sigma_fx * (-1.64485)) = 0.92 => sigma_fx = -ln(0.92)/1.64485.
sigma_fx = -np.log(0.92) / 1.64485  # ≈ 0.0507

# IR parameters: baseline r0 = 0.01 and sigma_ir = 0.0025/1.64485.
r0 = 0.01
sigma_ir = 0.0025 / 1.64485  # ≈ 0.00152

# Default hazard rate: 100bps per month → lambda_T = 0.01.
lambda_T = 0.01

# Define correlation matrices:
corr_scenarios = {
    "No Correlation": np.eye(3),
    "Mild Correlation": np.array([[1.0, 0.2, 0.2],
                                  [0.2, 1.0, 0.2],
                                  [0.2, 0.2, 1.0]]),
    "High Correlation": np.array([[1.0, 0.7, 0.7],
                                  [0.7, 1.0, 0.7],
                                  [0.7, 0.7, 1.0]]),
    "Very High Correlation": np.array([[1.0, 0.99, 0.99],
                                       [0.99, 1.0, 0.99],
                                       [0.99, 0.99, 1.0]])
}

# # Stress event thresholds:
# # --- Option A: Directional events ---
# fx_threshold_directional = 0.92  # FX ≤ 0.92 (drop >8%)
# ir_threshold_directional = r0 + 0.0025  # IR ≥ 0.0125 (increase >25bps)

# --- Option B: Absolute events ---
fx_threshold_absolute = 0.08  # |FX - 1| ≥ 0.08
ir_threshold_absolute = 0.0025  # |IR - r0| ≥ 0.0025

default_threshold = 1.0  # T ≤ 1 month

#########################################
# Step 5: Run Simulation and Compute Probabilities
#########################################
results = {}

for scenario_name, corr_matrix in corr_scenarios.items():
    copula = GaussianCopula(corr_matrix)
    simulation = MultiFactorSimulation(copula=copula, sigma_fx=sigma_fx, r0=r0, sigma_ir=sigma_ir, lambda_T=lambda_T,
                                       num_samples=num_samples)
    FX, IR, T = simulation.generate_joint_distribution()

    # Choose which stress event definition to use:
    # (Uncomment one of the two options below.)

    # Option A: Directional events
    # fx_stress = (FX <= fx_threshold_directional)
    # ir_stress = (IR >= ir_threshold_directional)

    # Option B: Absolute events
    fx_stress = (np.abs(FX - 1) >= fx_threshold_absolute)
    ir_stress = (np.abs(IR - r0) >= ir_threshold_absolute)

    # Default event: T ≤ 1
    default_event = (T <= default_threshold)

    # Joint stress: all three stress events occur.
    joint_event = fx_stress & ir_stress & default_event

    p_joint = np.mean(joint_event)
    p_default = np.mean(default_event)
    cond_prob = p_joint / p_default if p_default > 0 else np.nan

    results[scenario_name] = {
        "p_joint": p_joint,
        "p_default": p_default,
        "conditional_prob": cond_prob
    }

#########################################
# Step 6: Print the Results
#########################################
for scenario, vals in results.items():
    print(f"Scenario: {scenario}")
    print(f"  P(Joint Stress Event) = {vals['p_joint']:.2%}")
    print(f"  P(Default) = {vals['p_default']:.2%}")
    print(f"  P(Stress on FX & IR | Default) = {vals['conditional_prob']:.2%}")
    print("")
