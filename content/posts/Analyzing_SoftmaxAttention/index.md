+++
date = '2026-01-22T17:14:58+08:00'
draft = false
title = 'Scaled Dot-Product Attention as Entropic Optimal Transport'
math = true
unsafe = false
+++
> **TL;DR**:  The design of Softmax Attention has been more intuitive and engineering-driven, lacking a firm mathematical basis. This post presents a functional analysis perspective, framing the Softmax function as the exact solution to an entropy-regularized optimal transport problem. This opens up a new theoretical angle for designing attention mechanisms.

## Derivation: Scaled Dot-Product Attention as Entropic Optimal Transport

The Softmax function inherent in modern attention mechanisms is not an arbitrary heuristic. It represents the unique solution to an **entropy-regularized optimization problem**, balancing the maximization of similarity against the maximization of distributional uncertainty (entropy).

### 1. The Optimization Objective

Given a query vector $\boldsymbol{q}$ and a set of key vectors $\{\boldsymbol{k}_j\}$, we seek a probability distribution $\boldsymbol{p}$ (a transport plan) that minimizes the expected transport cost (negative similarity) subject to entropic regularization.

We define the minimization functional with temperature parameter $\tau$:
`$$ \min_{\boldsymbol{p}} \mathcal{J}(\boldsymbol{p}) = \underbrace{-\sum_{j=1}^m p_j (\boldsymbol{q}^T \boldsymbol{k}_j)}_{\text{Expected Cost}} - \underbrace{\tau H(\boldsymbol{p})}_{\text{Entropy Reg.}} $$`

Subject to the simplex constraint $\sum_{j=1}^m p_j = 1$, where $H(\boldsymbol{p}) = -\sum p_j \log p_j$ is the Shannon entropy.

### 2. Lagrangian Formulation

We employ the method of Lagrange multipliers to enforce the normalization constraint. Introducing $\lambda$, the Lagrangian is:

`$$
 \mathcal{L}(\boldsymbol{p}, \lambda) = -\sum_{j} p_j (\boldsymbol{q}^T \boldsymbol{k}_j) + \tau \sum_{j} p_j \log p_j + \lambda \left(1 - \sum_{j} p_j \right) 
$$`

### 3. First-Order Optimality Conditions

To find the stationary point, we compute the gradient with respect to $p_j$ and set it to zero (KKT condition):

$$ \frac{\partial \mathcal{L}}{\partial p_j} = -(\boldsymbol{q}^T \boldsymbol{k}_j) + \tau (1 + \log p_j) - \lambda = 0 $$

### 4. Solution Derivation

Rearranging terms to solve for $p_j$:

$$ \tau \log p_j = \boldsymbol{q}^T \boldsymbol{k}_j + \lambda - \tau $$

exponentiating yields:

$$ p_j = \exp\left( \frac{\boldsymbol{q}^T \boldsymbol{k}_j}{\tau} \right) \cdot \exp\left( \frac{\lambda - \tau}{\tau} \right) $$

The term $\exp((\lambda - \tau)/\tau)$ is constant for all $j$. Let this scaling factor be $1/Z$. By enforcing the constraint $\sum p_j = 1$, we derive the partition function $Z$:

$$ Z = \sum_{l=1}^m \exp\left( \frac{\boldsymbol{q}^T \boldsymbol{k}_l}{\tau} \right) $$

Substituting $Z$ back yields the canonical **Softmax** formulation:

`$$ p_j^\star = \text{Softmax}(\boldsymbol{q}, \boldsymbol{K})_j = \frac{\exp(\boldsymbol{q}^T \boldsymbol{k}_j / \tau)}{\sum_{l=1}^m \exp(\boldsymbol{q}^T \boldsymbol{k}_l / \tau)} $$`

## Reference
* Litman, E. (2025). *Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport*. arXiv preprint arXiv:2508.08369. Available at: https://doi.org/10.48550/arXiv.2508.08369.