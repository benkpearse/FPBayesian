import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# --- Sidebar Inputs ---
st.sidebar.header("Test Parameters")
p_A = st.sidebar.slider("Baseline conversion rate (p_A)", 0.001, 0.20, 0.05, step=0.001,
                         help="Conversion rate for your control variant (A), e.g., 5% means 0.05")
thresh = st.sidebar.slider("Posterior threshold (e.g., 0.95)", 0.5, 0.99, 0.95, step=0.01,
                           help="Confidence level to declare a winner — usually 0.95 or 0.99")
simulations = st.sidebar.slider("Simulations", 100, 2000, 500, step=100,
                                help="How many full A/B simulations to run — more is slower but more accurate")
samples = st.sidebar.slider("Posterior samples", 1000, 10000, 3500, step=500,
                            help="Number of random samples from the posterior Beta distributions")
n = st.sidebar.slider("Sample size per variant", 500, 200000, 35000, step=500,
                      help="Number of users (or sessions) tested in each variant arm")

# --- Simulation Function for False Positive ---
def simulate_false_positive(p_A, threshold, simulations, samples, n):
    alpha_prior, beta_prior = 1, 1
    false_positives = 0

    for _ in range(simulations):
        conv_A = np.random.binomial(n, p_A)
        conv_B = np.random.binomial(n, p_A)  # No true uplift

        post_A = beta(alpha_prior + conv_A, beta_prior + n - conv_A)
        post_B = beta(alpha_prior + conv_B, beta_prior + n - conv_B)

        samples_A = post_A.rvs(samples)
        samples_B = post_B.rvs(samples)

        if np.mean(samples_B > samples_A) > threshold:
            false_positives += 1

    return false_positives / simulations

# --- Run Simulation ---
fp_rate = simulate_false_positive(p_A, thresh, simulations, samples, n)

# --- Output ---
st.title("Bayesian False Positive Rate Estimator")

st.markdown("""
This app estimates the **false positive rate** of a Bayesian A/B test given:
- No true difference between variants (A = B)
- A posterior decision threshold (e.g., P(B > A) > 0.95)

You can adjust all parameters in the sidebar.
""")

st.write(f"**Baseline Conversion Rate:** {p_A:.2%}")
st.write(f"**Posterior Threshold:** {thresh:.2f}")
st.write(f"**Sample Size per Variant:** {n}")
st.write(f"**Estimated False Positive Rate:** {fp_rate:.2%}")

# --- Visual Aid ---
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(["False Positive Rate"], [fp_rate], color='salmon')
ax.axhline(0.05, color='red', linestyle='--', label='5% Threshold')
ax.set_ylim(0, max(fp_rate + 0.02, 0.1))
ax.set_ylabel("Rate")
ax.set_title("Estimated False Positive Rate")
ax.legend()
st.pyplot(fig)
