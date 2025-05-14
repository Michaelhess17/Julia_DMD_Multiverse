using Flux
using LinearAlgebra # For dot product if needed, though direct multiplication is fine

"""
    polynomial_basis(x, degree=2)

Generates a polynomial basis transformation of the input features `x`
up to the specified `degree`.

If `x` is a vector, it is treated as a single sample.
If `x` is a matrix, each column is treated as a sample.

The output for a single sample [x1, x2, ..., xn] includes:
[1, x1, x2, ..., xn, x1^2, x1*x2, ..., xn^2, ...]

For degree 2, the order is typically:
[1, x1, x2, ..., xn, x1^2, x2^2, ..., xn^2, x1*x2, x1*x3, ..., x_{n-1}*xn]
"""
function polynomial_basis_zygote(X::AbstractMatrix; degree::Int=2)
    n_features, n_samples = size(X)

    # Degree 0: Constant term
    constant_block = ones(1, n_samples)

    if degree == 0
        return constant_block
    end

    # Degree 1: Linear terms
    linear_block = X
    if degree == 1
        return vcat(constant_block, linear_block)
    end

    # Degree 2 and higher
    if degree >= 2
        # Squared terms: Element-wise square of the input matrix
        squared_block = X .^ 2

        # Interaction terms (x_i * x_j for i < j)
        # Generate interaction terms as a Vector of Vectors using a comprehension.
        # vcatting this Vector{Vector} will produce the desired matrix.
        interaction_terms_vecs = [X[i, :] .* X[j, :] for i in 1:n_features for j in (i + 1):n_features]

        # Concatenate all blocks. vcat is non-mutating.
        # Splatting (...) expands the interaction_terms_vecs into arguments for vcat.
        if degree == 2
             # vcat expects arguments to be of compatible dimensions for vertical stacking.
             # constant_block is 1xN
             # linear_block is n_features x N
             # squared_block is n_features x N
             # vcat(interaction_terms_vecs...) will produce a (num_interaction_terms) x N matrix
            #  @show size(constant_block), size(linear_block), size(squared_block), size(interaction_terms_vecs), size(vcat(interaction_terms_vecs...))
             return vcat(constant_block, linear_block, squared_block, hcat(interaction_terms_vecs...)')
        end
    end

    # Handle unsupported degrees
    error("Polynomial basis degree $degree not supported by this function. Only degrees 0, 1, and 2 are implemented.")
end

# function polynomial_basis(X::AbstractMatrix; degree::Int=2)
#     # Apply the transformation to each column (sample)
#     hcat([polynomial_basis_zygote(X[:, i], degree=degree) for i in 1:size(X, 2)]...)
# end

# --- Example Usage ---

# Single sample
# x_single = [1.0, 2.0]
# basis_single = polynomial_basis(x_single, degree=2)
# println("Polynomial basis for a single sample:")
# println(basis_single)
# # Expected output (order may vary slightly based on implementation details of interaction terms):
# # [1.0, 1.0, 2.0, 1.0, 4.0, 2.0] # 1, x1, x2, x1^2, x2^2, x1*x2

# println("-"^20)

# # Batch of samples (features in rows, samples in columns)
# X_batch = rand(2, 3) # 2 features, 3 samples
# basis_batch = polynomial_basis(X_batch, degree=2)
# println("Polynomial basis for a batch of samples:")
# println(basis_batch)
# # The output matrix will have size (number of basis functions) x (number of samples)
# # For 2 features and degree 2, number of basis functions = 1 (const) + 2 (linear) + 2 (squared) + 1 (interaction) = 6
# # Output size should be 6x3

# # Example of using it in a simple custom layer
struct PolynomialLayer
    degree::Int
end

Flux.@layer PolynomialLayer # This allows Flux to see the layer for parameters (though this layer has none)

(p::PolynomialLayer)(x) = polynomial_basis_zygote(x, degree=p.degree)

# # Create an instance of the custom layer
# poly_layer = PolynomialLayer(2)

# # Apply the layer to data
# output_single = poly_layer(x_single)
# output_batch = poly_layer(X_batch)

# println("\nOutput from PolynomialLayer (single sample):")
# println(output_single)

# println("\nOutput from PolynomialLayer (batch samples):")
# println(output_batch)