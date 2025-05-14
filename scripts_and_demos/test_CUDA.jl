using LinearAlgebra
using CUDA
using GPUArrays

PX = CuArray{Float32}(randn(1500, 500))
PY = CuArray{Float32}(randn(1500, 500))
W = CuArray{Float32}(ones(1500)./1500)
x_pts = collect(Float32, -1.5:0.01:1.5)
y_pts = collect(Float32, -1.5:0.01:1.5)
z_pts = CuArray([complex(a, b) for a in x_pts, b in y_pts])

# backend = get_backend(PX)


    
## compute the pseudospectrum
W_vec = vec(W) # Ensure W is a column vector
# In Julia, element-wise multiplication is .*
# `qr` by default computes the "thin" QR decomposition similar to MATLAB's "econ"
Q, R = qr(sqrt.(W_vec) .* PX)
Q = CuArray(Q)

C1 = (sqrt.(W_vec) .* PY) / R
L = C1' * C1
G = CuArray{eltype(PX)}(I, size(PX, 2), size(PX, 2)) # Identity matrix with correct type
A = Q' * C1

z_pts_vec = vec(z_pts) # Ensure z_pts is a column vector
LL = length(z_pts_vec)
RES = CuArray(zeros(Float32, LL))

function get_HZ_eig(z_pts_vec, RES, G, A, L)
    for I = 1:length(z_pts_vec)
        Hz = (complex.(L) .- z_pts_vec[I] * complex.(A')) .- (conj(z_pts_vec[I]) .* A) .+ (abs(z_pts_vec[I])^2 .* G)

        λs = eigvals(Hz)

            
        @inbounds RES[I] = sqrt(max(0.0f0, real(λs[1])))
    end
    nothing
end

@cuda get_HZ_eig(z_pts_vec, RES, G, A, L)

