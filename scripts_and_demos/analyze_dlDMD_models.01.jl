
using JLD2, LinearAlgebra, CairoMakie, KernelDensity
using StatsBase
using LaTeXStrings
using CSV, DataFrames
using Measures, StatsPlots


set_theme!(theme_latexfonts(font_size=14))

include("../../utils/statistics.jl")

results = load("../outputs/dlDMD_models_losses_eigs.jld2")
meta = CSV.read("/home/michael/Synology/Julia/data/all_human_data_metadata.csv", DataFrame)
models_states = results["models"]
eigenvalues = results["eigs"]

AB_inds = findall(meta[!, :lf_or_hf] .== "AB")
LF_inds = findall(meta[!, :lf_or_hf] .== "LF")
HF_inds = findall(meta[!, :lf_or_hf] .== "HF")
ST_inds = vcat(LF_inds, HF_inds)

dt = 0.01
model_inds = CartesianIndices(eigenvalues)

all_inds = vcat(AB_inds, ST_inds)
skip_inds = [ind for ind in all_inds if length(eigenvalues[ind]) == 1]
AB_inds = [ab for ab in AB_inds if ab ∉ skip_inds]
ST_inds = [st for st in ST_inds if st ∉ skip_inds]
AB_eigs = reduce(hcat, [eig for eig in eigenvalues[AB_inds] if length(eig) > 1])
ST_eigs = reduce(hcat, [eig for eig in eigenvalues[ST_inds] if length(eig) > 1])

all_eigs = hcat(AB_eigs, ST_eigs)

all_eigs_density = kde(abs.(all_eigs)[:])


peak_density = all_eigs_density.x[argmax(all_eigs_density.density)]
cutoff_density = peak_density - 1e-3

function get_var_explained_Fourier_eigs(model::Chain, y::AbstractArray, cutoff_density::AbstractFloat)
    enc, dec = Chain(model.layers[1:4]), Chain(model.layers[5:end])
    Ψ = enc(y)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    K = Ψ_plus * pinv(Ψ_minus)
    # F = svd(Ψ_minus)
    # K = F.U'*Ψ_plus*F.V*inv(Diagonal(F.S))
    E, V = eigen(K)

    k = V \ Ψ[:, 1]

    E_Diagonal = Matrix(Diagonal(E))
    E_Fourier = deepcopy(E_Diagonal)
    for ii in 1:size(E_Fourier, 1)
        if abs(E_Fourier[ii, ii]) < cutoff_density
            E_Fourier[ii, ii] = 0 + 0im
        end
    end

    A = V*E_Fourier*pinv(V)
    Ψ̂ = reduce(hcat, [real.(V*E_Fourier^(jj)*k) for jj in collect(1:size(Ψ_minus, 2))::Vector{Int}])

    residual = dec(Ψ̂ ) .- y[:, 2:end]
    var_explained = 1 - (var(residual)/var(y))
    return var_explained
end

function get_var_explained_Fourier_eigs(model::Chain, y::AbstractArray, n_eigs::Int)
    enc, dec = Chain(model.layers[1:4]), Chain(model.layers[5:end])
    Ψ = enc(y)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]
    _, _, _, ind, E, V = get_dmd_powers(model, y, true)
    E[1:n_eigs] .= 0.0+0.0im
    E_Diagonal = Matrix(Diagonal(E))

    A = V*E_Diagonal*pinv(V)
    k = V \ Ψ[ind, 1]
    Ψ̂ = reduce(hcat, [real.(V*E_Diagonal^(jj)*k) for jj in collect(1:size(Ψ_minus, 2))::Vector{Int}])

    residual = dec(Ψ̂ ) .- y[:, 2:end]
    var_explained = 1 - (var(residual)/var(y))
    return var_explained
end

function get_model(data_size::Int, layer_1_size::Int, layer_2_size::Int, latent_dim::Int, device)
    model = Chain(Dense(data_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_2_size, selu),
                    Dense(layer_2_size => latent_dim), # ) 

                    Dense(latent_dim => layer_2_size, selu),
                    Dense(layer_2_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => data_size)) 
    return model
end

function get_dmd_powers(m::Chain, y::AbstractArray{T}, return_eigs::Bool=false)
    enc, dec = Chain(m.layers[1:4]), Chain(m.layers[5:end])
    Ψ = enc(y)
    Ȳ = dec(Ψ)

    Ψ .-= mean(Ψ, dims=2)
    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    F = svd(Ψ_minus)
    # K = F.U'*Ψ_plus*F.V*inv(Diagonal(F.S))
    K = Ψ_plus * pinv(Ψ_minus)
    
    E, V = eigen(inv(Diagonal(sqrt.(F.S)))*K*Diagonal(sqrt.(F.S)))
    
    W = Diagonal(sqrt.(F.S))*V

    modes = Ψ_plus*F.V*inv(Diagonal(sqrt.(F.S)))*W

    ω = log.(E) ./ dt
    freqs = abs.(imag.(ω) / 2pi)
    powers = [norm(col)^2 for col in eachcol(modes)]

    if return_eigs
        ind = sortperm(powers)
        E, V = E[ind], V[:, ind]
        return freqs[ind], powers[ind], ω[ind], ind, E, V
    else
        return freqs, powers, ω
    end
end


T = Float32
numSubjects = 299

all_data = [Matrix(load_data(ii, T, cpu_device())) for ii in 1:numSubjects]
all_data = permutedims(reshape(reduce(hcat, all_data), size(all_data[1])..., length(all_data)), (3, 1, 2))
loaded_models = Vector{Chain}(undef, size(models))
for jj in 1:length(loaded_models)
    model = get_model(size(all_data[jj], 1), 128, 64, size(models[jj].layers[4].weight, 1), cpu_device())
    model = Flux.loadmodel!(model, models[jj])
    loaded_models[jj] = model
end

var_exp = [get_var_explained_Fourier_eigs(model, y, cutoff_density) for (ii, (model, y)) in enumerate(zip(loaded_models, all_data)) if ii ∉ skip_inds ]

transient_eigs = [eig[abs.(eig) .< cutoff_density] for eig in eachcol(all_eigs)]
mean_transient_eigs = [median(abs.(eig)) for eig in transient_eigs]

fig = Figure(fontsize=18)
ax = Axis(fig[1, 1], xlabel="Proportion of Variance Explained by Unitary Eigenmodes", ylabel=L"\bar{\lambda}_{\textrm{transient}}")
CairoMakie.scatter!(ax, var_exp[1:length(AB_inds)], mean_transient_eigs[1:length(AB_inds)], label="Tied", color=:dodgerblue)
CairoMakie.scatter!(ax, var_exp[length(AB_inds):end], mean_transient_eigs[length(AB_inds):end], label="Split", color=:crimson)
axislegend(position=:lb)
save("figures/percent_var_explained_vs_median_transient_eigs_64_dims_mice.png", fig)


freqs_powers_ω = [get_dmd_powers(m, y) for (ii, (m, y)) in enumerate(zip(loaded_models2, all_data2)) if ii ∉ skip_inds]
freqs = reduce(hcat, [reverse(freq_power[1]) for freq_power in freqs_powers_ω])
powers = reduce(hcat, [reverse(freq_power[2]) for freq_power in freqs_powers_ω])
ωs = reduce(hcat, [reverse(freq_power[3]) for freq_power in freqs_powers_ω])
Es = reduce(hcat, [reverse(freq_power[5]) for freq_power in freqs_powers_ω])
Vs = permutedims(stack([freq_power[6][:, end:-1:1] for freq_power in freqs_powers_ω], dims=3), (3, 1, 2))

for idx in 1:size(powers, 2)
    mask = freqs[:, idx] .== 0.0
    powers[mask, idx] .= 0.0
end 

fig = Figure()
use_AB = [12]
use_ST = [52]
ax = Axis(fig[1, 1], xlabel=L"\Re{\frac{\log{\lambda}}{\Delta t}}", ylabel="Freqency [Hz]")
[CairoMakie.scatter!(ax, real.(ωs[:, ind]), freqs[:, ind], markersize=20*powers[:, ind]./maximum(powers[:, ind]), color=:dodgerblue, label="AB") for ind in use_AB]
[CairoMakie.scatter!(ax, real.(ωs[:, ind]), freqs[:, ind], markersize=20*powers[:, ind]./maximum(powers[:, ind]), color=:crimson, label="ST") for ind in use_ST]
# CairoMakie.xlims!(ax, -2, 0.2)
axislegend(merge=true)
save("tmp2.png", fig)



N = size(Es, 2)
max_rank = 24
# Separate data for each group
group1_eigenvalues = Es'[AB_inds, 1:2:max_rank]
group2_eigenvalues = Es'[ST_inds, 1:2:max_rank]

# Calculate the mean absolute eigenvalue for each rank in each group
# We use `abs` because the request is for "absolute values"
mean_abs_group1 = mean(log.(real.(group1_eigenvalues)) ./ dt, dims=1)[1, :]
mean_abs_group2 = mean(log.(real.(group2_eigenvalues)) ./ dt, dims=1)[1, :]
sem_group1 = std(log.(real.(group1_eigenvalues)), dims=1)[1, :]# / sqrt(n)
sem_group2 = std(log.(real.(group2_eigenvalues)), dims=1)[1, :]# / sqrt(N - n)


df = DataFrame(
    Rank = Int64[], # For "G1", "G2" etc. or "Rank 1", "Rank 2"
    MeanAbsEigenvalue = Float64[],
    Group = String[],
    yErr = Float64[]
)

# Populate the DataFrame
jj = 1
for i in 1:2:max_rank
    @show jj
    push!(df, (jj, mean_abs_group1[jj], "AB", sem_group1[jj]))
    push!(df, (jj, mean_abs_group2[jj], "ST", sem_group2[jj]))
    jj += 1
end

println("\nDataFrame for plotting (first 5 rows):")
display(first(df, 5))

# --- 4. Create the Grouped Bar Plot ---

p = groupedbar(df.Rank, df.MeanAbsEigenvalue,
        group = df.Group, # This tells groupedbar how to group the bars
        ylabel = L"\Re{\frac{\log{\lambda}}{\Delta t}}",
        xlabel = L"\lambda \quad \textrm{Rank}",
        legend = :bottomleft,
        yerr= df.yErr,
        # palette = :Set1,
        palette = [:dodgerblue, :crimson],
        lw = 0.5,
        framestyle = :box,
        grid = :y,
        # size = (900, 600),
        yformatter = :plain, # Prevents scientific notation on y-axis if values are small
        margin=5mm
        )

# You can save the plot
savefig(p, "eigenvalue_comparison_barplot.png")
