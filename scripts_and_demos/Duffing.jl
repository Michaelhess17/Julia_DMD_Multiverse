# using DrWatson
# quickactivate("julia_env")
using DifferentialEquations
using LinearAlgebra
using Statistics
using Clustering
using Random
using CairoMakie

include("utils/resDMD.jl")

# Define ODE
ODEFUN!(dy, y, p, t) = begin
    dy[1] = y[2]
    dy[2] = y[1] - y[1]^3
end

# van der Pol oscillator
ODEFUN2!(dy, y, μ, t) = begin
    dy[1] = y[2]
    dy[2] = -y[1] + μ * y[2] * (1 - y[1]^2)
end

# Parameters
M1 = 10^2
M2 = 150
delta_t = 0.05
N = 100
PHI(r) = exp.(-r)

# Grid for pseudospectra
x_pts = -1.2:0.025:1.2
y_pts = -1.2:0.025:1.2
v = 10 .^ (-2:0.1:1)

X = Array{Float64}(undef, 2, M2, M1)
Y = Array{Float64}(undef, 2, M2, M1)

μ = 1.0
Threads.@threads for jj in 1:M1
    Y0 = (rand(2) .- 0.5) .* 4
    tspan = (0.0, delta_t * (3 + M2))
    prob = ODEProblem(ODEFUN2!, Y0, tspan, μ)
    sol = solve(prob, Tsit5(), saveat=delta_t)
    Y1 = Array(sol)  # shape: (2, M2 + 4)

    # We skip the first few points, as in original slicing
    X[:, :, jj] = Y1[:, [1; 3:M2+1]]     # size: (2, M2)
    Y[:, :, jj] = Y1[:, 3:M2+2]          # size: (2, M2)
end

X = reshape(X, 2, :)
Y = reshape(Y, 2, :)


M = M1 * M2

# Compute scaling
X_mean = mean(X, dims=2)
d = mean(norm.(eachcol(X .- X_mean)))

# KMeans clustering to get centers
data_for_kmeans = hcat(X, Y)
kmeans_result = kmeans(data_for_kmeans, N)
C = kmeans_result.centers'  # Nx2 matrix

# Build PX and PY matrices
PX = zeros(M, N)
PY = zeros(M, N)

for j in 1:N
    R_X = sqrt.((X[1,:] .- C[j,1]).^2 .+ (X[2,:] .- C[j,2]).^2)
    PX[:,j] = PHI(R_X ./ d)
    
    R_Y = sqrt.((Y[1,:] .- C[j,1]).^2 .+ (Y[2,:] .- C[j,2]).^2)
    PY[:,j] = PHI(R_Y ./ d)
end

# Solve Koopman operator
K = PX \ PY
eigvals_K, eigvecs_K = eigen(K)
LAM = eigvals_K

# Residuals
res = [norm(PY * v - PX * v * λ) / norm(PX * v) for (v, λ) in zip(eachcol(eigvecs_K), LAM)]

# Plot eigenvalues colored by residuals
f = Figure()
ax = Axis(f[1,1], aspect = 1, limits = (-1, 1, -1, 1), xlabel = "Re(λ)", ylabel = "Im(λ)", title = "Koopman Eigenvalues Colored by Residuals")
scatter!(ax, real(LAM), imag(LAM), color=res, colormap=:turbo)


# --- Pseudospectra computation ---

# Form grid
# z_pts = kron(x_pts, ones(length(y_pts))) .+ im * kron(ones(length(x_pts)), y_pts)
z_pts_mat = [complex(a, b) for a = x_pts, b = y_pts]
z_pts = vec(z_pts_mat)  # flatten

# Compute pseudospectrum
RES, _, _ = KoopPseudoSpecQR(PX, PY, ones(size(PX, 1))/M, z_pts, z_pts2=ComplexF64[], reg_param=1e-14, progress=true)
RES = reshape(RES, (length(x_pts), length(y_pts)))

# Plot pseudospectrum

f = Figure()

ax, hm = contourf(f[1, 1][1, 1],
         x_pts,
         y_pts,
         log10.(max.(minimum(v), real.(RES))),
         levels=log10.(v),
         colormap=:grays, 
         axis = (aspect = 1, limits = (minimum(x_pts), maximum(x_pts), minimum(y_pts), maximum(y_pts)), 
         xlabel = L"\mathrm{Re}(\lambda)", ylabel = L"\mathrm{Im}(\lambda)", title = L"\mathrm{Sp}_\epsilon(\mathcal{K}),\quad N = %$N"))

Colorbar(f[1, 1][1, 2], hm, label = L"\tau(\lambda)")

save("tmp.png", f)