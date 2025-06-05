using JLD2, LinearAlgebra, CairoMakie, KernelDensity
using StatsBase
using LaTeXStrings

set_theme!(theme_latexfonts())

include("../../utils/statistics.jl")

data = load("../outputs/dlDMD_models_losses_eigs.jld2")
eigs = data["eigs"]

AB_inds = 1:30
ST_inds = 31:72

AB_eigs = reduce(hcat, [eig for eig in eigs[AB_inds] if length(eig) > 1])
ST_eigs = reduce(hcat, [eig for eig in eigs[ST_inds] if length(eig) > 1])

skip_inds = [ind for ind in vcat(AB_inds, ST_inds) if length(eigs[ind]) == 1]

all_eigs = hcat(reduce(hcat, AB_eigs), reduce(hcat, ST_eigs))

all_eigs_density = kde(abs.(all_eigs)[:])

f = Figure()
ax = Axis(f[1, 1], xlabel=L"| \lambda |", ylabel="Density", title="All eigenvalues (broken down by population)")
bins = 50
edges = StatsBase.fit(Histogram, abs.(all_eigs)[:]; nbins=bins).edges[1]
CairoMakie.hist!(ax, abs.(AB_eigs)[:], bins=edges, label="AB", color=(:dodgerblue, 0.8), normalization=:pdf)
CairoMakie.hist!(ax, abs.(ST_eigs)[:], bins=edges, label="ST", color=(:crimson, 0.8), normalization=:pdf)
CairoMakie.plot!(ax, all_eigs_density, color=:black, linestyle=:dash)
axislegend()

save("figures/abs_eigenvalues_histogram.png", f)

peak_density = all_eigs_density.x[argmax(all_eigs_density.density)]
cutoff_density = peak_density - 5e-3

unit_circle_eigs = [eig[abs.(eig) .>= cutoff_density] for eig in eachcol(all_eigs)]
transient_eigs = [eig[abs.(eig) .< cutoff_density] for eig in eachcol(all_eigs)]

n_unit_circle_eigs = [length(eig) for eig in unit_circle_eigs]
n_transient_eigs = [length(eig) for eig in transient_eigs]

inds_mapping = setdiff(vcat(AB_inds, ST_inds), skip_inds)

n_unit_circle_eigs_AB = [n_unit_circle_eig for (ii, n_unit_circle_eig) in enumerate(n_unit_circle_eigs) if inds_mapping[ii] in AB_inds]
n_unit_circle_eigs_ST = [n_unit_circle_eig for (ii, n_unit_circle_eig) in enumerate(n_unit_circle_eigs) if inds_mapping[ii] in ST_inds]

f = Figure()
ax = Axis(f[1, 1], xlabel="# Eigenvalues on Unit Circle", ylabel="Count", title="\"Control\" modes by population")
bins = 20
edges = StatsBase.fit(Histogram, n_unit_circle_eigs; nbins=bins).edges[1]
CairoMakie.hist!(ax, n_unit_circle_eigs_AB, bins=edges, label="AB", color=(:dodgerblue, 0.8))
CairoMakie.hist!(ax, n_unit_circle_eigs_ST, bins=edges, label="ST", color=(:crimson, 0.8))
axislegend()

save("figures/n_unit_circle_eigs.png", f)

f = Figure()
ax = Axis(f[1, 1], xlabel=L"| \lambda |", ylabel="Density", title="Unit-circle eigenvalues")
bins = 50
AB_sample = reduce(vcat, [abs.(unit_circle_eig) for (ii, unit_circle_eig) in enumerate(unit_circle_eigs) if inds_mapping[ii] in AB_inds])
ST_sample = reduce(vcat, [abs.(unit_circle_eig) for (ii, unit_circle_eig) in enumerate(unit_circle_eigs) if inds_mapping[ii] in ST_inds])

edges = StatsBase.fit(Histogram, vcat(AB_sample, ST_sample); nbins=bins).edges[1]
ab = CairoMakie.hist!(ax, AB_sample, bins=edges, label="AB", color=(:dodgerblue, 0.8), normalization=:pdf)
st = CairoMakie.hist!(ax, ST_sample, bins=edges, label="ST", color=(:crimson, 0.8), normalization=:pdf)
axislegend()
save("figures/unit_circle_eigenvalues_AB_vs_ST.png", f)


f = Figure()
ax = Axis(f[1, 1], xlabel=L"\textrm{Bootstrap Mean } | \lambda |", ylabel="Density", title="Bootstrap means of unit-circle eigenvalues")
bins = 50
AB_sample = get_bootstrapped_sample(reduce(vcat, [abs.(unit_circle_eig) for (ii, unit_circle_eig) in enumerate(unit_circle_eigs) if inds_mapping[ii] in AB_inds]))
ST_sample = get_bootstrapped_sample(reduce(vcat, [abs.(unit_circle_eig) for (ii, unit_circle_eig) in enumerate(unit_circle_eigs) if inds_mapping[ii] in ST_inds]))

edges = StatsBase.fit(Histogram, vcat(AB_sample, ST_sample); nbins=bins).edges[1]
ab = CairoMakie.hist!(ax, AB_sample, bins=edges, label="AB", color=(:dodgerblue, 0.8), normalization=:pdf)
st = CairoMakie.hist!(ax, ST_sample, bins=edges, label="ST", color=(:crimson, 0.8), normalization=:pdf)
axislegend(ax)

save("figures/unit_circle_eigenvalues_AB_vs_ST_bootstrapped.png", f)


f = Figure()
ax = Axis(f[1, 1], xlabel=L"| \lambda |", ylabel="Density", title="Transient eigenvalues")
bins = 50
AB_sample = reduce(vcat, [abs.(transient_eig) for (ii, transient_eig) in enumerate(transient_eigs) if inds_mapping[ii] in AB_inds])
ST_sample = reduce(vcat, [abs.(transient_eig) for (ii, transient_eig) in enumerate(transient_eigs) if inds_mapping[ii] in ST_inds])

edges = StatsBase.fit(Histogram, vcat(AB_sample, ST_sample); nbins=bins).edges[1]
ab = CairoMakie.hist!(ax, AB_sample, bins=edges, label="AB", color=(:dodgerblue, 0.8), normalization=:pdf)
st = CairoMakie.hist!(ax, ST_sample, bins=edges, label="ST", color=(:crimson, 0.8), normalization=:pdf)
axislegend(ax)
save("figures/transient_eigenvalues_AB_vs_ST.png", f)


f = Figure()
ax = Axis(f[1, 1], xlabel=L"\textrm{Bootstrap Mean } | \lambda |", ylabel="Density", title="Bootstrap means of transient eigenvalues")
bins = 50
AB_sample = get_bootstrapped_sample(reduce(vcat, [abs.(transient_eig) for (ii, transient_eig) in enumerate(transient_eigs) if inds_mapping[ii] in AB_inds]))
ST_sample = get_bootstrapped_sample(reduce(vcat, [abs.(transient_eig) for (ii, transient_eig) in enumerate(transient_eigs) if inds_mapping[ii] in ST_inds]))

edges = StatsBase.fit(Histogram, vcat(AB_sample, ST_sample); nbins=bins).edges[1]
ab = CairoMakie.hist!(ax, AB_sample, bins=edges, label="AB", color=(:dodgerblue, 0.8), normalization=:pdf)
st = CairoMakie.hist!(ax, ST_sample, bins=edges, label="ST", color=(:crimson, 0.8), normalization=:pdf)
axislegend(ax)

save("figures/transient_eigenvalues_AB_vs_ST_bootstrapped.png", f)


mean_transient_eigs = [mean(abs.(eig)) for eig in transient_eigs]
max_transient_eigs = [maximum(abs.(eig)) for eig in transient_eigs]


f = Figure()
ax = Axis(f[1, 1], xlabel="# Eigenvalues on Unit Circle", ylabel=L"Mean | \lambda_{transient} |", title="Number of \"control\" modes vs. mean stability timescale")

AB_sample = [mean_transient_eig for (ii, mean_transient_eig) in enumerate(mean_transient_eigs) if inds_mapping[ii] in AB_inds]
ST_sample = [mean_transient_eig for (ii, mean_transient_eig) in enumerate(mean_transient_eigs) if inds_mapping[ii] in ST_inds]

CairoMakie.scatter!(ax, n_unit_circle_eigs_AB, AB_sample, label="AB", color=:dodgerblue, alpha=0.7, markersize=5)
CairoMakie.scatter!(ax, n_unit_circle_eigs_ST, ST_sample, label="ST", color=:crimson, alpha=0.7, markersize=5)

axislegend(ax)

save("figures/n_unit_circle_eigs_vs_mean_transient_eigs.png", f)
