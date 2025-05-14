using LinearAlgebra

include("exactDMD.jl")

function compressedSensingDMD(X::AbstractArray{T}, C::AbstractArray{T}, B::AbstractArray{T}) where T <: Number
    Y = X[:, 2:end]
    r = size(X, 1)
    K_c = exactDMD(X, r)
    _, Σ_c, _ = truncatedSVD(K_c, r)
    ϕ_c, Λ_c = getExactDMDModes(K_c, Y, Σ_c)
    ϕ_s = applyL1(ϕ_c)
    ϕ = B*ϕ_s
    return ϕ, Λ_c
end

function applyL1(ϕ_c::AbstractArray{T}) where T <: Number
    # CoSaMP algorithm (Needell & Tropp, 2009)
    
end