# NOT tested or ready yet!

using LinearAlgebra

include("exactDMD.jl")

function rangeFinder(X::AbstractArray{T, 2}, r::Int, p::Int, q::Int) where T <: Number
    d, M = size(X)
    Ω = randn(M, (r+p))
    Z = X*Ω
    for jj = 1:q
        FQ = qr(Z)
        Q = Matrix(FQ.Q)
        FC = qr(X'*Q)
        C = Matrix(FC.Q)
        Z = X*C
    end
    FZ = qr(Z)
    Z = Matrix(FZ.Q)[:, 1:(r+p)]
    return Z
end

function randomizedDMD(X::AbstractArray{T, 2}, r::Int, p::Int, q::Int)::Tuple{AbstractArray{S, 2}, AbstractArray{S, 2}} where T <: Number, S <: Number
    X, Y = X[:, 1:(end-1)], X[:, 2:end]
    d, M = size(X)
    Q = rangeFinder(X, r, p, q)
    X_c = Q'*X
    Y_c = Q'*Y
    K_c = exactDMD(X, r)
    _, Σ_c, _ = truncatedSVD(K_c, r)
    Φ_c, Λ_c = getExactDMDModes(K_c, Y, Σ_c)
    ϕ = Q*Φ_c
    return ϕ, Λ_c
end   