+++
date = '2025-12-25T17:14:58+08:00'
draft = false
title = 'Analyzing Sparse Attention Approximation With SNR'
math = true
unsafe = false
+++

> **TL;DR**: **Linear Attention** fails when attention distributions are "peaky" (high dynamic range) because it acts as a low-pass filter. **Sparse Attention** approximates Softmax by keeping an **Active Set ($\mathcal{A}_i$)** of high scores and discarding a **Pruned Set ($\mathcal{A}_i^c$)**. The approximation error is determined by the Residual Weight ($\epsilon_i$)—the probability mass lost in the pruned set. We model this using **Signal-to-Noise Ratio (SNR)**: High SNR (peaky distribution) yields low error, while low SNR (flat distribution) leads to significant bias. The method is exponentially sensitive to Recall Failure: missing even a single high-scoring token (where $s_{\max}^c \approx m_i$) causes the SNR to collapse and the error to spike.


# Review: The approximation limiation of Linear Attention
In the previous post, we established that Linear Attention operates as a first-order Taylor approximation of the Softmax function:
$$
e^x \approx 1 + x
$$
Applying this to attention scores $s_{ij} = \frac{q_i k_j^T}{\sqrt{d}}$ centered around their maximum $m_i$:
$$
\text{Softmax}(s_{ij}) \approx \frac{1 + (s_{ij} - m_i)}{\sum_j (1 + s_{ij} - m_i)}
$$

**Key Finding**: The approximation error is governed by the Lagrange remainder term $R_1(x) = O(\frac{x^2}{2})$. Thus, the fidelity of Linear Attention depends inversely on the **dynamic range** of the scores for a given query:
$$
\Delta_i = \max_j(s_{ij}) - \min_j(s_{ij})
$$
When $\Delta_i$ is large (implying a "peaky" or sparse distribution), the linear approximation diverges significantly from Softmax. Linear Attention effectively acts as a low-pass filter, smoothing out the sharp distinctions that Softmax creates.

# 1. Analyzing Sparse Attention's Approximation Error.
## 1.1 Formulation: A Set-Theoretic View

To analyze the error rigorously, we first define Sparse Attention as a domain restriction operation on the standard Softmax.

Let $\Omega = \{1, \dots, N\}$ be the full set of key indices. For a specific query $i$, Sparse Attention partitions $\Omega$ into two disjoint sets based on a selection policy $\pi$:

1.  **The Active Set ($\mathcal{A}_i$)**: The subset of indices containing the "heavy hitters" (significant scores).
2.  **The Pruned Set ($\mathcal{A}_i^c$)**: The complement set containing negligible scores.

$$
\Omega = \mathcal{A}_i \cup \mathcal{A}_i^c, \quad \mathcal{A}_i \cap \mathcal{A}_i^c = \emptyset
$$

Standard Softmax Attention computes the expectation over the full measure space $\Omega$. For numerical stability, we subtract the maximum score $m_i = \max_{j \in \Omega}(s_{ij})$:

`$$
\begin{aligned}
O_i^{\text{full}} &= \frac{\sum_{j \in \mathcal{A}_i} e^{s_{ij}-m_i} v_j + \sum_{j \in \mathcal{A}_i^c} e^{s_{ij}-m_i} v_j}{\sum_{j \in \mathcal{A}_i} e^{s_{ij}-m_i} + \sum_{j \in \mathcal{A}_i^c} e^{s_{ij}-m_i}} \\
&= \frac{N(\mathcal{A}_i) + N(\mathcal{A}_i^c)}{D(\mathcal{A}_i) + D(\mathcal{A}_i^c)}
\end{aligned}
$$`
where $N(\cdot)$ and $D(\cdot)$ represent the numerator and denominator components for the respective sets.

Sparse Attention approximates this by assuming the probability weight in $\mathcal{A}_i^c$ is zero. It truncates the summation to the Active Set:

`$$
O_i^{\text{sparse}} = \frac{\sum_{j \in \mathcal{A}_i} e^{s_{ij}-m_i} v_j}{\sum_{j \in \mathcal{A}_i} e^{s_{ij}-m_i}} = \frac{N(\mathcal{A}_i)}{D(\mathcal{A}_i)}
$$`

## 1.2 Approximation Error of Sparse Attention

The approximation error $\mathcal{E}_i$ is the difference between the full attention output and the sparse approximation:

`$$
\mathcal{E}_i = O_i^{\text{full}} - O_i^{\text{sparse}} = \frac{N(\mathcal{A}_i) + N(\mathcal{A}_i^c)}{D(\mathcal{A}_i) + D(\mathcal{A}_i^c)} - \frac{N(\mathcal{A}_i)}{D(\mathcal{A}_i)}
$$`

By finding a common denominator, we can expand this term to reveal the structure of the error:

`$$
\begin{aligned}
\mathcal{E}_i &= \frac{\left(N(\mathcal{A}_i) + N(\mathcal{A}_i^c)\right)D(\mathcal{A}_i) - N(\mathcal{A}_i)\left(D(\mathcal{A}_i) + D(\mathcal{A}_i^c)\right)}{D(\mathcal{A}_i)\left(D(\mathcal{A}_i) + D(\mathcal{A}_i^c)\right)} \\
&= \frac{N(\mathcal{A}_i)D(\mathcal{A}_i) + N(\mathcal{A}_i^c)D(\mathcal{A}_i) - N(\mathcal{A}_i)D(\mathcal{A}_i) - N(\mathcal{A}_i)D(\mathcal{A}_i^c)}{D(\mathcal{A}_i) D_{\text{total}}} \\
&= \frac{N(\mathcal{A}_i^c)D(\mathcal{A}_i) - N(\mathcal{A}_i)D(\mathcal{A}_i^c)}{D(\mathcal{A}_i) D_{\text{total}}}
\end{aligned}
$$`

This can be rearranged into a more intuitive form:

`$$
\mathcal{E}_i = \underbrace{\frac{D(\mathcal{A}_i^c)}{D_{\text{total}}}}_{\text{Residual Weight } (\epsilon_i)} \cdot \left( \underbrace{\frac{N(\mathcal{A}_i^c)}{D(\mathcal{A}_i^c)}}_{\text{Context form Pruned Set}} - \underbrace{\frac{N(\mathcal{A}_i)}{D(\mathcal{A}_i)}}_{\text{Context from Active Set}} \right)
$$`

This equation tells us that the error is the product of two factors:
1.  **The Residual Weight ($\epsilon_i$)**: The total probability weight assigned to the pruned tokens. If the attention distribution is sharp, this term is small.
2.  **The Context Divergence**: The difference between the weighted average value vector of the pruned set versus the active set.

Therefore, Sparse Attention fails if either the pruned set contains significant probability weight (selection failure) or if the pruned values are vastly different from the active values (contextual loss).


## 1.3 Signal-Noise Ratio(SNR) View
We can reinterpret the error term using a Signal-to-Noise Ratio (SNR) analogy. Let the "Signal" be the contribution from the Active Set ($\mathcal{A}_i$) and the "Noise" be the contribution from the Pruned Set ($\mathcal{A}_i^c$).

The full attention output is a convex combination of the signal and the noise:

`$$
O_i^{\text{full}} = (1 - \epsilon_i) \cdot O_i^{\text{sparse}} + \epsilon_i \cdot O_i^{\text{noise}}
$$`

where:
*   $O_i^{\text{sparse}} = \frac{N(\mathcal{A}_i)}{D(\mathcal{A}_i)}$ is the output derived purely from the selected top-k tokens.
*   $O_i^{\text{noise}} = \frac{N(\mathcal{A}_i^c)}{D(\mathcal{A}_i^c)}$ is the output derived from the pruned tokens.
*  `$\epsilon_i = \frac{D(\mathcal{A}_i^c)}{D_{\text{total}}}$` is the weight of the noise (the residual probability weight).

From this perspective, the goal of any sparse attention mechanism is to maximize the SNR by minimizing $\epsilon_i$.

*   **High SNR (Peaky Distribution)**: If the attention distribution is very sharp, almost all probability weight is concentrated in $\mathcal{A}_i$. Here, $\epsilon_i \to 0$, and $O_i^{\text{full}} \approx O_i^{\text{sparse}}$. Sparse attention works perfectly.
*   **Low SNR (Flat Distribution)**: If the attention is diffuse (e.g., uniform), significant weight leaks into $\mathcal{A}_i^c$. Here, $\epsilon_i$ is large, and ignoring the "noise" term $O_i^{\text{noise}}$ introduces significant bias.

## 1.4 Analysis of the error term $\epsilon_i$

To understand when Sparse Attention fails, we must analyze the behavior of the residual weight term $\epsilon_i$. Recall that:

`$$
\epsilon_i = \frac{D(\mathcal{A}_i^c)}{D_{\text{total}}} = \frac{D(\mathcal{A}_i^c)}{D(\mathcal{A}_i) + D(\mathcal{A}_i^c)}
$$`

By dividing the numerator and denominator by $D(\mathcal{A}_i^c)$, we can rewrite this in a form that isolates the ratio of the "Active" weight to the "Pruned" weight:

$$
\epsilon_i = \frac{1}{1 + \frac{D(\mathcal{A}_i)}{D(\mathcal{A}_i^c)}}
$$

Let $\rho_i = \frac{D(\mathcal{A}_i)}{D(\mathcal{A}_i^c)}$ be the **Signal-to-Noise Ratio (SNR)** of the probability weightes. The error term $\epsilon_i$ is strictly decreasing with respect to $\rho_i$. To minimize error ($\epsilon_i \to 0$), we need $\rho_i \to \infty$. Let's analyze the components of $\rho_i$:

`$$
\begin{aligned}
\rho_i &= \frac{\sum_{j \in \mathcal{A}_i} e^{s_{ij}-m_i}}{\sum_{j \in \mathcal{A}_i^c} e^{s_{ij}-m_i}} \\
&\ge \frac{1 + (|\mathcal{A}_i| - 1)e^{s_k - m_i}}{|\mathcal{A}_i^c| \cdot e^{s_{k+1} - m_i}} \\
&= \frac{1}{|\mathcal{A}_i^c|} \left( \frac{1}{e^{s_{k+1} - m_i}} + (|\mathcal{A}_i| - 1)\frac{e^{s_k - m_i}}{e^{s_{k+1} - m_i}} \right) \\
&= \frac{1}{|\mathcal{A}_i^c|} \left( \underbrace{e^{m_i - s_{k+1}}}_{\text{Peak-to-Noise Gap}} + (|\mathcal{A}_i| - 1)\underbrace{e^{s_k - s_{k+1}}}_{\text{Boundary Gap}} \right)
\end{aligned}
$$`

In this lower bound, the **Peak-to-Noise Term** plays the dominant role. To maximize $\rho_i$, we can:

1.  **Increase the Numerator**: Widen the gap $m_i - s_{k+1}$ and increase $|\mathcal{A}_i|$ (e.g., by increasing $K$).
2.  **Decrease the Denominator**: Reduce $|\mathcal{A}_i^c|$ by filtering out irrelevant tokens.


## 1.5 Sparse Attention's error under limited block granularity

In the derivation above, we assumed an ideal selection policy where `$\mathcal{A}_i$` contains the strict top-$k$ scores, making $s_{k+1}$ the upper bound of the pruned set. However, in practice—especially with block-sparse methods or approximate nearest neighbor search—we cannot guarantee strict ordering. We might miss a high-scoring token or include a lower-scoring one.

To formalize this, we must relax the bound on the denominator. We replace the strict boundary score $s_{k+1}$ with the actual maximum score found within the pruned set, denoted as `$s_{\max}^c = \max_{j \in \mathcal{A}_i^c}(s_{ij})$`.

Consequently, the range of $s_{\max}^c$ expands. In the worst-case scenario (a failure to retrieve the true maximum), $s_{\max}^c$ could be as large as the global maximum $m_i$ (i.e., $s_0$). In the best case (perfect retrieval), it is $s_{k+1}$.

$$
s_{k+1} \le s_{\max}^c \le s_0
$$

Updating our SNR lower bound with this relaxation:

`$$
\rho_i \ge \frac{1}{|\mathcal{A}_i^c|} \cdot e^{\Delta}, \quad \text{where } \Delta = m_i - s_{\max}^c
$$`


This formulation highlights the critical risk of sparse attention: **Recall Failure**. If the selection policy fails to retrieve the true peak (i.e., $s_{\max}^c \approx m_i$), the exponent approaches 0, $\rho_i$ collapses, and the error $\epsilon_i$ becomes significant. The approximation quality is exponentially sensitive to the gap between the captured maximum and the missed maximum.



## Reference
* Xiao G.(2025) Blog. “[Statistics behind Block Sparse Attention](https://hanlab.mit.edu/blog/block-sparse-attn-stats)”
