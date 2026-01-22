# NeuralBradleyTerry-Ligue1

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Math](https://img.shields.io/badge/Theory-Bradley--Terry-purple)

This repository implements a Neural Bradley-Terry model to estimate the latent strength of European football teams and predict match outcomes. Unlike classical statistical methods that assign a static parameter to each team, this approach learns a non-linear mapping function from team features (financial, historical, tactical) to a latent strength score. This formulation effectively solves the Cold-Start Problem for newly promoted teams, as Paris FC in the season 2025/2026, and captures the complex relationship between squad market value and on-pitch performance.

---

## Mathematical Framework

### 1. The Classical Bradley-Terry Model

The standard Bradley-Terry model postulates that the probability of team $i$ defeating team $j$ depends on their respective latent strengths $\theta_i$ and $\theta_j$:

$$P(i \succ j) = \frac{\exp(\theta_i)}{\exp(\theta_i) + \exp(\theta_j)} = \sigma(\theta_i - \theta_j)$$

Where $\sigma$ is the sigmoid function.

**Limitation:** This model treats $\theta_i$ as a fixed parameter to be learned from history. If team $i$ has no history (e.g., a promoted team), $\theta_i$ cannot be estimated accurately.

### 2. The Neural Extension 

We replace the static parameter $\theta_i$ with a parameterized function $f_\phi$ (a Neural Network) that acts on a feature vector $\mathbf{x}_i$:

$$\theta_i = f_\phi(\mathbf{x}_i)$$

Where:
* $\mathbf{x}_i \in \mathbb{R}^d$ is a vector of structural and tactical features.
* $f_\phi$ is a Multi-Layer Perceptron (MLP) with weights $\phi$.

The probability of a home win for team $i$ against team $j$ becomes:

$$P(\text{Home}) = \sigma(f_\phi(\mathbf{x}_i) - f_\phi(\mathbf{x}_j) + \alpha)$$

Where $\alpha$ represents the home field advantage.

### 3. Optimization & Time Decay

We minimize the Binary Cross-Entropy (BCE) loss. To account for concept drift, the fact that team dynamics change over years, we implement a quadratic time-decay weighting scheme:

$$\mathcal{L}(\phi) = - \sum_{k=1}^{N} w_k \cdot [y_k \log(\hat{p}_k) + (1-y_k) \log(1-\hat{p}_k)]$$

Where $w_k \propto (t_k - t_{min})^2$ ensures that recent matches contribute more to the gradient updates than older ones.

---

## Feature Engineering & Priors

The model's power lies in its input features $\mathbf{x}_i$, designed to disentangle "luck" from "potential":

| Feature Category | Variable | Theoretical Justification |
| :--- | :--- | :--- |
| **Structural** | `Log_Market_Value` | Proxy for intrinsic squad quality. Solves the cold start problem. |
| **Historical** | `Pythagorean_Exp` | $Goals^2 / (Goals^2 + Conceded^2)$. A better predictor of future points than actual points. |
| **Tactical** | `Avg_Shots_Target` | Measures offensive creation independent of finishing variance. |
| **Playstyle** | `Home_Dependency` | Captures teams that overperform at home but struggle away. |

*Note: To prevent data leakage, all tactical features are (N-1) priors (aggregates from the previous season).*

## Probabilistic Forecasting

Unlike simple classification, this model outputs full probability distributions. We handle the draw outcome by defining a "Draw Window" around the probability equilibrium:

* $\hat{y} = \text{Home}$ if $P(\text{diff}) > 0.5 + \tau$
* $\hat{y} = \text{Away}$ if $P(\text{diff}) < 0.5 - \tau$
* $\hat{y} = \text{Draw}$ otherwise

Where $\tau$ is calibrated via Grid Search to match the historical draw frequency of Ligue 1 (~26%).

---

## Usage

### Prerequisites
* Python 3.8+
* PyTorch, Pandas, NumPy, Matplotlib
