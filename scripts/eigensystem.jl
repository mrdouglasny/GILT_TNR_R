# MODULE: eigensystem.jl
# USAGE: julia +1.10.4 --project=. scripts/eigensystem.jl <config>
# INPUTS: ekrgilttrnr/configs/eigensystem/<config>.toml
# OUTPUTS: ekrgilttrnr/data/eigensystem/<config>.data
# DESCRIPTION: Compute RG Jacobian eigenvalues for 2D Ising model at criticality
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (arXiv:2408.10312), ekrgilttrnr/scripts/newton.jl
# NEW IN THIS SCRIPT: Wrapper script to compute and analyze the eigenspectrum
#                     of the RG Jacobian at the fixed point found by Newton's method.
#
# Quick test run (< 5 min):
#   julia +1.10.4 --project=. scripts/eigensystem.jl quick
#
# Production run (~30 min):
#   julia +1.10.4 --project=. scripts/eigensystem.jl default
#
# Expected results (2D Ising, c=1/2 CFT):
#   - Energy operator ε: λ ≈ 2.0 (Δ = 1)
#   - Stress tensor T, T̄: λ ≈ ±1.0 (Δ = 2, marginal)
#   - Magnetization σ: λ ≈ 3.668 (Δ = 1/8)
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312)
# NEW IN THIS SCRIPT: Adapted for config-driven runs with ProjectUtils integration
#                     and output to ekrgilttrnr/data/ following PROTOCOL.md.
#
# RUNTIME: ~5 min (quick), ~30 min (default χ=30). Scales as O(χ⁶).
#
# sections:
# - EXPERIMENT INIT
# - DISCREE GAUGE FIXING MATRICES, RECURSION DEPTH, RMATRICES
# - FUNCTIONS
# - EIGENVALUES


################################################
# section: EXPERIMENT INIT
################################################

# Load ProjectUtils for config handling (from rg/src/julia/)
include("../../src/julia/ProjectUtils.jl")
using .ProjectUtils

# Note: Originally developed with Julia 1.10.4, works with 1.12.3 on cluster

include("../src/Tools.jl");
include("../src/GaugeFixing.jl");
include("../src/KrylovTechnical.jl");

# Setup with config (loads TOML, creates paths)
# base_dir="ekrgilttrnr" since scripts run from rg/ on cluster
config_name, config, paths = setup_with_config("eigensystem", ARGS; base_dir="ekrgilttrnr")

# Extract parameters with defaults
chi = get(config, "chi", 30)
gilt_eps = get(config, "gilt_eps", 6e-6)
cg_eps = get(config, "cg_eps", 1e-10)
rotate = get(config, "rotate", false)
relT = get(config, "relT", 0.0)
Jratio = get(config, "Jratio", 1.0)
number_of_initial_steps = get(config, "number_of_initial_steps", 23)
ord = get(config, "ord", 2)
stp = get(config, "stp", 1e-4)
N = get(config, "N", 10)
krylovdim = get(config, "krylovdim", 30)
verbosity = get(config, "verbosity", 1)
Z2_odd_sector = get(config, "Z2_odd_sector", true)
freeze_R = get(config, "freeze_R", false)
path_to_tensor = get(config, "path_to_tensor", "none")

gilt_pars = Dict(
	"gilt_eps" => gilt_eps,
	"cg_chis" => collect(1:chi),
	"cg_eps" => cg_eps,
	"verbosity" => 0,
	"rotate" => rotate,
)

if path_to_tensor == "none"
	if relT == 0 # then we search in critical_temperatures/ to find critical relT
		# if case is not found in critical_temperatures we throw an exception
		search_tol = 1.0e-10
		relT_low, relT_high, relT = find_critical_temperature(chi, gilt_pars, search_tol, Jratio)
		if abs(relT) < 1.e-10
			throw("Critical relT was not found in critical_temperatures/")
		end
	end

	initialA_pars = Dict("relT" => relT, "Jratio" => Jratio)

	A_crit_approximation = trajectory(initialA_pars, number_of_initial_steps, gilt_pars)["A"][end]
else
	tensor_data = deserialize(path_to_tensor)
	A_crit_approximation = tensor_data["A"]
end

A_crit_approximation_Z2_broken = A_crit_approximation.to_ndarray();

A_crit_approximation, _ = fix_continuous_gauge(A_crit_approximation);
A_crit_approximation, accepted_elements, _ = fix_discrete_gauge(A_crit_approximation; tol = 1e-7);
A_crit_approximation /= A_crit_approximation.norm();


A_crit_approximation_JU = py_to_ju(A_crit_approximation);

ZH = diagm(vcat(ones(chi ÷ 2), -ones(chi ÷ 2)))
ZV = diagm(vcat(ones(chi ÷ 2), -ones(chi ÷ 2)))

A_crit_approximation_Z2_broken, GH, GV, SH, SV = fix_continuous_gauge(A_crit_approximation_Z2_broken);
A_crit_approximation_Z2_broken, accepted_elements_Z2_broken, _, _ = fix_discrete_gauge(A_crit_approximation_Z2_broken);
normalize!(A_crit_approximation_Z2_broken);

ZH = GH' * ZH * GH
ZV = GV' * ZV * GV

#####################################################################
# section: DISCREE GAUGE FIXING MATRICES, RECURSION DEPTH, RMATRICES
#####################################################################


py"""
def gilttnr_step_broken(A,log_fact,pars):
	A=Tensor.from_ndarray(A)
	if "Rmatrices" in pars:
		for key,value in  pars["Rmatrices"].items():
			pars["Rmatrices"][key]=Tensor.from_ndarray(value)
	return gilttnr_step(A,log_fact,pars)
"""


if Z2_odd_sector
	Atmp, _ = py"gilttnr_step_broken"(A_crit_approximation_Z2_broken, 0.0, gilt_pars)
	Atmp, _ = fix_continuous_gauge(Atmp)
	Atmp, _, H, V = fix_discrete_gauge(Atmp, accepted_elements_Z2_broken)
else
	Atmp, _ = py"gilttnr_step"(A_crit_approximation, 0.0, gilt_pars)
	Atmp, _ = fix_continuous_gauge(Atmp)
	Atmp, _, H, V = fix_discrete_gauge(Atmp, accepted_elements)
end;


Rmatrices = py"Rmatrices"
tmp = py"depth_dictionary"


if path_to_tensor == "none"
	recursion_depth = Dict(
		"S" => tmp[(1, "S")],
		"N" => tmp[(1, "N")],
		"E" => tmp[(1, "E")],
		"W" => tmp[(1, "W")],
	)
else
	recursion_depth = tensor_data["recursion_depth"]
end

if freeze_R
	gilt_pars = Dict(
		"gilt_eps" => gilt_eps,
		"cg_chis" => collect(1:chi),
		"cg_eps" => cg_eps,
		"verbosity" => 0,
		"bond_repetitions" => 2,
		"Rmatrices" => Rmatrices,
		"rotate" => rotate,
	)
else
	gilt_pars = Dict(
		"gilt_eps" => gilt_eps,
		"cg_chis" => collect(1:chi),
		"cg_eps" => cg_eps,
		"verbosity" => 0,
		"bond_repetitions" => 2,
		"recursion_depth" => recursion_depth,
		"rotate" => rotate,
	)
end


################################################
# section: FUNCTIONS
################################################

function gilt(A, pars)
	A = ju_to_py(A)
	A, _, _ = py"gilttnr_step"(A, 0.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A, accepted_elements)
	#A = ncon([A, H.conj(), V.conj(), H, V], [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
	A /= A.norm()
	return py_to_ju(A)
end

function gilt(A::Array, pars)
	A, _, _ = py"gilttnr_step_broken"(A, 0.0, pars)
	A, _ = fix_continuous_gauge(A)
	A, _ = fix_discrete_gauge(A, accepted_elements_Z2_broken)
	#A = ncon([A, H, V, H, V], [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
	normalize!(A)
	return A
end

function dgilt(δA)
	return df(x -> gilt(x, gilt_pars), A_crit_approximation_JU, δA; stp = stp, order = ord)
end

function dgilt(δA::Array)
	return df(x -> gilt(x, gilt_pars), A_crit_approximation_Z2_broken, δA; stp = stp, order = ord)
end

#################################################
# section: EIGENVALUES
#################################################

if Z2_odd_sector
	initial_vector = normalize!(2 .* rand(chi, chi, chi, chi) .- 1)
else
	initial_vector = py_to_ju(random_Z2tens(A_crit_approximation))
end;

res = eigsolve(dgilt, initial_vector, N, :LM; verbosity = verbosity, issymmetric = false, ishermitian = false, krylovdim = krylovdim, maxiter = 200);

if freeze_R
	result = Dict(
		"A" => A_crit_approximation,
		"eigensystem" => res,
		"bond_repetitions" => 2,
		"Rmatrices" => Rmatrices,
	)
else
	result = Dict(
		"A" => A_crit_approximation,
		"eigensystem" => res,
		"bond_repetitions" => 2,
		"recursion_depth" => recursion_depth,
	)
end


# Use paths from ProjectUtils (already created directories)
data_filename = replace(paths["data"], ".csv" => ".data")
serialize(data_filename, result)

# Print results
println("\n" * "=" ^ 60)
println("EIGENVALUES for Ising model (config: $config_name)")
println("  Jratio=$Jratio, chi=$chi, gilt_eps=$gilt_eps")
println("=" ^ 60)

println("\nExpected (2D Ising, c=1/2 CFT):")
println("  σ (magnetization): λ ≈ 3.668 (Δ = 1/8)")
println("  ε (energy):        λ ≈ 2.0   (Δ = 1)")
println("  T (stress):        λ ≈ 1.0   (Δ = 2, marginal)")
println()

println("Computed eigenvalues:")
println("-" ^ 50)
for i in range(1, length(res[1]))
	val = res[1][i]
	z2_parity = parity(res[2][i], ZH, ZV)
	println("  λ_$i = $(round(real(val), digits=4)) + $(round(imag(val), digits=4))i  |  Z₂ = $z2_parity")
end

# Append results to log file (ProjectUtils already created it)
log_filename = paths["log"]
open(log_filename, "a") do f
	println(f, "\n" * "=" ^ 50)
	println(f, "Ising Jacobian Eigenvalue Computation")
	println(f, "Config: $config_name")
	println(f, "=" ^ 50)
	println(f, "Parameters:")
	println(f, "  Jratio = $Jratio")
	println(f, "  χ = $chi, gilt_eps = $gilt_eps, cg_eps = $cg_eps")
	println(f, "  Z2_odd_sector = $Z2_odd_sector, freeze_R = $freeze_R")
	println(f, "  number_of_initial_steps = $number_of_initial_steps")
	println(f, "")
	println(f, "Eigenvalues:")
	for i in range(1, length(res[1]))
		val = res[1][i]
		z2_parity = parity(res[2][i], ZH, ZV)
		println(f, "  λ_$i = $(round(real(val), digits=6)) + $(round(imag(val), digits=6))i  |  Z₂ = $z2_parity")
	end
	println(f, "")
	println(f, "Expected (2D Ising universality):")
	println(f, "  λ_σ ≈ 3.668, λ_ε ≈ 2.0, λ_T ≈ 1.0")
end

println("\nResults saved to: $data_filename")
println("Log saved to: $log_filename")

#=
# print complex eigenvalue
for i in range(1, length(res[1]))
	val = res[1][i]
	println(Jratio, " : ", val, " | evalonly evnum=", i, " Z2 q.n.=", parity(res[2][i], ZH, ZV), " chi=", chi, " gilt_eps=", gilt_eps, " cg_eps=", cg_eps)
end


# print real part of eigenvalue
for i in range(1, length(res[1]))
	val = real(res[1][i])
	println(Jratio, "  ", val, " | evalreal evnum=", i, " Z2 q.n.=", parity(res[2][i], ZH, ZV), " chi=", chi, " gilt_eps=", gilt_eps, " cg_eps=", cg_eps)
end

# print abs of eigenvalue
for i in range(1, length(res[1]))
	val = abs(res[1][i])
	println(Jratio, "  ", val, " | evalabs evnum=", i, " Z2 q.n.=", parity(res[2][i], ZH, ZV), " chi=", chi, " gilt_eps=", gilt_eps, " cg_eps=", cg_eps)
end

# print log of abs of eigenvalue
for i in range(1, length(res[1]))
	val = log(abs(res[1][i]))
	println(Jratio, "  ", val, " | evallog evnum=", i, " Z2 q.n.=", parity(res[2][i], ZH, ZV), " chi=", chi, " gilt_eps=", gilt_eps, " cg_eps=", cg_eps)
end
=#