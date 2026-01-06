# MODULE: check_ising_residual.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/check_ising_residual.jl
# INPUTS: ekrgilttrnr/critical_temperatures/*.data (saved critical temperature)
# OUTPUTS: stdout (residual ||RG(A) - A|| at fixed point)
# DESCRIPTION: Check Ising fixed point quality by computing residual ||RG(A) - A||.
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312)
# NEW IN THIS SCRIPT: Quick diagnostic to check if RG trajectory has converged
#                     to fixed point by measuring residual after N steps.
#
# RUNTIME: ~5 min at χ=30. Quick diagnostic script.

using PyCall
using LinearAlgebra

include("../src/Tools.jl")
include("../src/GaugeFixing.jl")

const chi = 30  # Use chi=30 to match saved critical temp
const gilt_pars = Dict("gilt_eps" => 6e-6, "cg_chis" => collect(1:chi), "cg_eps" => 1e-10, "verbosity" => 0, "rotate" => false)

# Look up saved critical temperature (relT=0 causes div by zero)
_, _, relT = find_critical_temperature(chi, gilt_pars, 1e-10, 1.0)
println("Using critical relT = $relT")
const initialA_pars = Dict("relT" => relT, "Jratio" => 1.0)

println("Ising fixed point quality at criticality (relT=0)")
println("="^50)

for n_steps in [10, 15, 20, 23, 30]
    traj = trajectory(initialA_pars, n_steps, gilt_pars)
    A = traj["A"][end]

    getitem = pyimport("operator").getitem
    result = pycall(py"gilttnr_step", PyObject, A, 0.0, gilt_pars)
    A_next = pycall(getitem, PyObject, result, 0)

    A_gf, _, _, _, _ = fix_continuous_gauge(A)
    A_gf, _, _, _ = fix_discrete_gauge(A_gf)
    A_gf = A_gf / A_gf.norm()

    A_next_gf, _, _, _, _ = fix_continuous_gauge(A_next)
    A_next_gf, _, _, _ = fix_discrete_gauge(A_next_gf)
    A_next_gf = A_next_gf / A_next_gf.norm()

    diff = (A_gf - A_next_gf).norm()
    println("n=$n_steps: ||RG(A)-A|| = $(round(diff, digits=6)), shape=$(A.shape)")
end
