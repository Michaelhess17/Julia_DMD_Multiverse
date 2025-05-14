# Not tested or ready yet!!

function applyFunctions(Ψ::Tuple{Vararg{Function}}, PX::AbstractArray{T, 2}) where T <: Number
    M, N = size(PX)
    Ψ_X = Array{T, 2}(undef, M, N*length(Ψ))
    for (i, ψ) in enumerate(Ψ)
        for j in 1:N
            Ψ_X[:, (i-1)*N+j] = ψ(PX[:, j])
        end
    end
    return Ψ_X
end

function extendedDMD(X::AbstractArray{T, 2}, Ψ::Tuple{Vararg{Function}}, W::AbstractArray{T, 1}=ones(size(X, 1)/size(X, 1))) where T <: Number
    X, Y = X[:, 1:end-1], X[:, 2:end]
    Ψ_X = applyFunctions(Ψ, X)
    Ψ_Y = applyFunctions(Ψ, Y)

    D = sqrt.(Diagonal(W))

    K = pinv(D*Ψ_X)*D*Ψ_Y

    Λ, V = eigen(K)
    
    return Λ, V
end