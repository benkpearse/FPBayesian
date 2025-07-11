import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt

# Seeded RNG for reproducibility
rng = np.random.default_rng(42)

# --- Sidebar Inputs ---
st.sidebar.header("Test Parameters")
p_A = st.sidebar.number_input("Baseline conversion rate (p_A)", min_value=0.001, max_value=0.99, value=0.05, step=0.001, format="%.3f")
thresh = st.sidebar.slider("Posterior threshold (e.g., 0.95)", 0.5, 0.99, 0.95, step=0.01)
simulations = st.sidebar.slider("Simulations", 100, 20000, 3000, step=100)  # Increased default for stability
samples = st.sidebar.slider("Posterior samples", 1000, 100000, 3500, step=500)

# --- Prior Inputs ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Prior Settings")
alpha_prior = st.sidebar.number_input("Alpha (prior successes)", min_value=0.1, value=1.0, step=0.1)
beta_prior = st.sidebar.number_input("Beta (prior failures)", min_value=0.1, value=1.0, step=0.1)

# --- Sample Size Controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("📉 Sample Size Settings")
plot_range = st.sidebar.checkbox("Plot FPR vs. sample size?")
smoothing = st.sidebar.checkbox("Smooth the curve (moving average)", value=True)

if plot_range:
    min_n = st.sidebar.number_input("Min sample size", 100, 100000, 5000, step=100)
    max_n = st.sidebar.number_input("Max sample size", min_n + 100, 200000, 50000, step=100)
    step_n = st.sidebar.number_input("Step size", 100, 20000, 5000, step=100)
else:
    n = st.sidebar.number_input("Sample size per variant", min_value=100, value=35000, step=100)

# --- Simulation Function ---
def simulate_false_positive(p_A, threshold, simulations, samples, n, alpha_prior, beta_prior):
    false_positives = 0
    for _ in range(simulations):
        conv_A = rng.binomial(n, p_A)
        conv_B = rng.binomial(n, p_A)

        post_A = beta(alpha_prior + conv_A, beta_prior + n - conv_A)
        post_B = beta(alpha_prior + conv_B, beta_prior + n - conv_B)

        samples_A = post_A.rvs(samples, random_state=rng)
        samples_B = post_B.rvs(samples, random_state=rng)

        if np.mean(samples_B > samples_A) > threshold:
            false_positives += 1

    return false_positives / simulations

# --- Main Title ---
st.title("🧪 Bayesian False Positive Rate Estimator")

st.markdown("""
Estimate how often a Bayesian A/B test might **falsely declare a winner** (B > A) when there's **no true difference**.
""")

# --- Simulation Logic ---
if plot_range:
    st.subheader("📉 False Positive Rate vs. Sample Size")
    fpr_results = []
    sizes = list(range(min_n, max_n + 1, step_n))

    with st.spinner("Running simulations..."):
        for curr_n in sizes:
            fpr = simulate_false_positive(p_A, thresh, simulations, samples, curr_n, alpha_prior, beta_prior)
            fpr_results.append(fpr)

    # Optional smoothing
    if smoothing:
        fpr_plot = pd.Series(fpr_results).rolling(window=3, min_periods=1, center=True).mean()
    else:
        fpr_plot = fpr_results

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, fpr_plot, marker='o')
    ax.axhline(0.05, color='red', linestyle='--', label='5% Benchmark')
    ax.set_xlabel("Sample Size per Variant")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("False Positive Rate vs. Sample Size")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown("### FPR values:")
    for size, rate in zip(sizes, fpr_results):
        st.write(f"Sample Size {size}: False Positive Rate = {rate:.2%}")

else:
    st.subheader("🔍 False Positive Rate Estimate (Single Sample Size)")
    fp_rate = simulate_false_positive(p_A, thresh, simulations, samples, n, alpha_prior, beta_prior)

    st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
    st.write(f"**Posterior Threshold:** {thresh:.2f}")
    st.write(f"**Sample Size per Variant:** {n}")
    st.write(f"**Alpha Prior:** {alpha_prior:.1f}")
    st.write(f"**Beta Prior:** {beta_prior:.1f}")
    st.success(f"**Estimated False Positive Rate:** {fp_rate:.2%}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["False Positive Rate"], [fp_rate], color='salmon')
    ax.axhline(0.05, color='red', linestyle='--', label='5% Benchmark')
    ax.set_ylim(0, max(fp_rate + 0.02, 0.1))
    ax.set_ylabel("Rate")
    ax.set_title("Estimated False Positive Rate")
    ax.legend()
    st.pyplot(fig)

# --- Misinterpretation Warning ---
st.markdown("""
<details>
<summary><strong>🧠 Understanding Bayesian False Positives</strong></summary>

Unlike frequentist tests, Bayesian A/B tests **do not control a fixed Type I error (α = 0.05)** by default.

- The **false positive rate can vary** with:
  - Sample size
  - Prior beliefs (Alpha and Beta)
  - Posterior decision threshold (e.g., 0.95)
- It is **not guaranteed to stay below 5%**, even with a 95% posterior threshold.

This app helps you **empirically test** and justify your Bayesian rules.

</details>
""", unsafe_allow_html=True)


