# **Copula-Based Joint Simulation with Heston & Stochastic Hazard Rate**

## **1️⃣ Problem Definition**
We model the **joint distribution of a default event and stock price**, where:
1. **Stock price follows a stochastic process (Heston Model)**.
2. **Default event is determined by a hazard rate process (Deterministic or CIR Model)**.
3. **Dependence is introduced via a copula function**.

---

## **2️⃣ Step 1: Define the Joint Distribution Using Sklar’s Theorem**
According to **Sklar’s Theorem**, any **joint cumulative distribution function (CDF)** $F_{X,Y}(x, y)$ can be decomposed into:

$$
F_{X,Y}(x, y) = C_{\theta} \left( F_X(x), F_Y(y) \right)
$$

where:
- $C_{\theta}(\cdot, \cdot)$ is a **copula function** parameterized by $\theta$, which models dependence.
- $F_X(x)$ and $F_Y(y)$ are the **marginal CDFs** of the two variables (default time and stock price).
- If $F_X$ and $F_Y$ are continuous, $C_{\theta}$ is unique.

This means that **instead of modeling the joint distribution directly, we can model**:
4. The **marginals** $F_X(x)$, $F_Y(y)$.
5. The **copula function** $C_{\theta}(u, v)$, which introduces dependency.

---

## **3️⃣ Step 2: Marginal Distributions**

### **3.1 Stock Price: Heston Model**
Instead of assuming a **lognormal stock price**, we use the **Heston model**:

$$
dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t^S
$$

$$
dV_t = \kappa (\theta_t - V_t) dt + \sigma_v \sqrt{V_t} dW_t^V
$$

where:
- $S_t$ is the stock price,
- $V_t$ is the variance process,
- $W_t^S$, $W_t^V$ are correlated Wiener processes with correlation $\rho$.

If $\theta_t$ is time-dependent, no closed-form mean/variance exists, so we use **numerical simulation**.

---

### **3.2 Default Process: Hazard Rate**
We assume the **default event** follows an **intensity process**:

#### **(i) Deterministic Hazard Rate**
$$
\lambda(t) = f(t)
$$
where $f(t)$ is a given function of time.

#### **(ii) Stochastic Hazard Rate (CIR Process)**
$$
d\lambda_t = \kappa_\lambda (\theta_\lambda - \lambda_t) dt + \sigma_\lambda \sqrt{\lambda_t} dW_t^\lambda
$$

where $W_t^\lambda$ is a Brownian motion.

If $\lambda_t$ is stochastic, we **numerically simulate** the process.

---

## **4️⃣ Step 3: Transforming Marginals to Uniform Space**
For any variable $X$ with a known **CDF** $F_X(x)$, we transform:

$$
U_X = F_X(X) \quad \text{(Maps to uniform } [0,1] \text{)}
$$

For variables with **no closed-form CDF** (e.g., Heston), we use the **empirical CDF**.

---

## **5️⃣ Step 4: Apply Copula for Dependence**
6. **Sample independent uniform variables**:
   $$
   U_1, U_2 \sim U(0,1)
   $$
7. **Introduce dependency using a copula**:
   $$
   U_2' = C_{\theta}^{-1}(U_1, U_2)
   $$
8. **Transform back to the original marginals**:
   $$
   X = F_X^{-1}(U_1), \quad Y = F_Y^{-1}(U_2')
   $$

For **numerically simulated marginals**, we use the **empirical inverse CDF**.

---

## **6️⃣ Step 5: Mathematically Expressing Different Copulas**
Each copula provides a different transformation $C_{\theta}^{-1}(U_X, U_Y)$.

### **6.1 Gaussian Copula**
Gaussian Copula introduces **correlation** in normal space:

- Convert to standard normal:
  $$
  Z_X = \Phi^{-1}(U_X), \quad Z_Y = \Phi^{-1}(U_Y)
  $$
- Apply correlation matrix $R$:
  $$
  (Z_X', Z_Y') = L \cdot (Z_X, Z_Y)
  $$
  where $L$ is the **Cholesky decomposition** of $R$.
- Convert back:
  $$
  V_X = \Phi(Z_X'), \quad V_Y = \Phi(Z_Y')
  $$

### **6.2 Student-t Copula**
Student-t copula works similarly but uses **t-distributions** instead:

$$
T_X = t_{\nu}^{-1}(U_X), \quad T_Y = t_{\nu}^{-1}(U_Y)
$$

where $t_{\nu}^{-1}$ is the **inverse Student-t CDF**.

### **6.3 Clayton Copula**
Clayton Copula is **lower tail dependent**:

$$
V_Y = \left( U_X^{-\theta} + U_Y^{-\theta} - 1 \right)^{-1/\theta}
$$

---

## **7️⃣ Step 6: Estimating the Joint Distribution**
Once we have transformed samples $(X, Y)$, we estimate the **joint empirical density**:

$$
\hat{f}(x, y) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - X_i, y - Y_i)
$$

where $K_h$ is a **kernel density estimator**.

Alternatively, use **2D histograms** or **3D surface plots** to visualize.

---

## **8️⃣ Step 7: Summary of Mathematical Process**
9. **Generate independent samples:** $U_X, U_Y \sim U(0,1)$.
10. **Transform to original space using CDFs:** $U_X = F_X(X)$, $U_Y = F_Y(Y)$.
11. **Apply copula transformation:** $(V_X, V_Y) = C_{\theta}^{-1}(U_X, U_Y)$.
12. **Convert back to real values:** $X = F_X^{-1}(V_X), Y = F_Y^{-1}(V_Y)$.
13. **Estimate joint empirical density** $\hat{f}(x,y)$.

---

## **9️⃣ Step 8: How This Relates to Object-Oriented Code**
| **Mathematical Step** | **Code Class/Method** |
|----------------------|----------------------|
| Generate $U(0,1)$ samples | `CopulaSimulation.generate_samples()` |
| Convert to uniform space | `BaseCopula.sample(U)` |
| Apply copula transformation | `GaussianCopula.sample()`, `StudentTCopula.sample()`, etc. |
| Convert back to original space | `CopulaSimulation.generate_joint_distribution()` |
| Compute empirical density | `plot_heatmap()`, `plot_3D_surface()` |
