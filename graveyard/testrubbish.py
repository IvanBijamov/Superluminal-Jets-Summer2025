import pytensor
import pytensor.tensor as pt
import numpy as np

# --- 1. Define Inputs and Constants ---

# Your input matrix of w_t values.
# Let's create a dummy 1000x1 matrix as you described ("a thousand values").
# The shape could be (1000,) or (1000, 1), etc. Broadcasting will handle it.
wt_matrix = pt.as_tensor_variable(np.random.randn(1000, 1), name="wt_matrix")

# Constants from your formula
m = pt.as_tensor_variable(3.0, name="m")  # Number of std deviations for the range
sigma_w = pt.as_tensor_variable(0.5, name="sigma_w")  # The std deviation σ_w
n_steps = 20  # Number of steps in the sum ("twenty iterations")

# The step size Δw. Calculated from the total range and number of steps.
# Total range is (wt + m*sigma) - (wt - m*sigma) = 2*m*sigma
delta_w = (2 * m * sigma_w) / n_steps


# Placeholder for the function C_w(w).
# You must replace this with your actual implementation.
# It must be composed of PyTensor operations.
def Cw(w):
    # For example, let's say C_w(w) is just 1.0 or some simple function.
    return 1.0


# --- 2. Create the 'w' vectors for EACH 'wt' using Broadcasting ---

# First, create a single, normalized vector for the summation range.
# This represents the steps from -m*sigma_w to +m*sigma_w.
w_range_vector = pt.linspace(-m * sigma_w, m * sigma_w, n_steps)  # Shape: (n_steps,)

# Now, create the full 3D grid for `w` and broadcasted `wt`.
# wt_matrix shape: (1000, 1) -> Reshape to (1000, 1, 1)
# w_range_vector shape: (n_steps,) -> Reshape to (1, 1, n_steps)
# The result `w` will have shape (1000, 1, n_steps)
w = wt_matrix[:, :, None] + w_range_vector[None, None, :]

# Broadcast wt_matrix to match the shape of `w` for the calculation.
wt_bcast = wt_matrix[:, :, None]


# --- 3. Calculate the Term Inside the Summation (Vectorized) ---

# All of these operations are applied element-wise on the (1000, 1, n_steps) grid.
w_minus_wt = w - wt_bcast

# Denominator from the formula: sqrt(2π) * σ_w^3
denominator = pt.sqrt(2 * np.pi) * (sigma_w**3)

# Exponential term: e^(-(w-wt)² / 2σ_w²)
exp_term = pt.exp(-(w_minus_wt**2) / (2 * sigma_w**2))

# The C_w(w) term. Our function is applied element-wise to the `w` grid.
Cw_values = Cw(w)

# Combine everything to get the full term being summed (the "summand").
summand = (w_minus_wt / denominator) * exp_term * Cw_values


# --- 4. Perform the Summation over 'w' ---

# We sum along the last axis (axis=2), which corresponds to the n_steps of `w`.
# The result collapses the grid back to the shape of wt_matrix: (1000, 1)
sum_over_w = pt.sum(summand, axis=2)

# Finally, multiply by Δw to get the matrix of P_obs results
P_obs_matrix = sum_over_w * delta_w


# --- 5. Calculate the Final Total Sum ---

# As requested, sum all the computed P_obs values.
total_P_obs_sum = pt.sum(P_obs_matrix)


# --- Execute and Print Results ---
print("--- Calculating P_obs based on the formula ---")

# .eval() will compile and run the entire graph.
p_obs_values = P_obs_matrix.eval()
total_sum_value = total_P_obs_sum.eval()

print(f"\nShape of the resulting P_obs matrix: {p_obs_values.shape}")
print(f"First 5 values of P_obs:\n{p_obs_values[:5].flatten()}")
print(f"\nFinal total sum of all P_obs values: {total_sum_value}")
