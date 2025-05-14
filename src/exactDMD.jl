function truncatedSVD(A::AbstractMatrix{T}, r::Int) where T <: Number
    U, S, Vt = svd(A)
    V = Vt'
    return U[:, 1:r], Diagonal(S[1:r]), V[:, 1:r]
end

function exactDMD(X::AbstractMatrix{T}, r::Int) where T <: Number
    X, Y = X[:, 1:end-1], X[:, 2:end]
    U, S, V = truncatedSVD(X, r)
    K_dmd = U'*Y*V*inv(S)
    return K_dmd
end

function getExactDMDModes(K::AbstractMatrix{S},  Y::AbstractMatrix{T}, Σ::AbstractMatrix{T})::Tuple{AbstractMatrix{S}, AbstractMatrix{S}} where T <: Number, S <: Complex
    λ, W = eigen(K)
    Λ = Diagonal(λ)
    ϕ = Y*V*inv(Σ)*W
    return ϕ, Λ
end

function getProjectedDMDModes(U::AbstractMatrix{T}, K::AbstractMatrix{S})::Tuple{AbstractMatrix{S}, AbstractMatrix{S}} where T <: Number, S <: Complex
    λ, W = eigen(K)
    Λ = Diagonal(λ)
    ϕ_r = U'*W
    return ϕ_r, Λ
end

# K_dmd * x_r^{n} = x_r^{n+1} where x_r is U'*x