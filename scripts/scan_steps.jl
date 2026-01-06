
using Serialization
using LinearAlgebra
using KrylovKit
using TOML

# Load ProjectUtils
include("../../src/julia/ProjectUtils.jl")
using .ProjectUtils

# Load tools
include("../src/Tools.jl")
include("../src/Phi4Tools.jl")
include("../src/GaugeFixing.jl")
include("../src/Phi4GaugeFixing.jl")
include("../src/KrylovTechnical.jl")
include("../src/NumDifferentiation.jl")
using .NumDifferentiation

# Parameters
config_path = "ekrgilttrnr/configs/phi4_eigensystem/quick.toml"
config = TOML.parsefile(config_path)

chi = config["chi"]
gilt_eps = config["gilt_eps"]
cg_eps = config["cg_eps"]
rotate = config["rotate"]
lam = config["lam"]
kappa = config["kappa"]
K = config["K"]
D = config["D"]
mu_sq = config["mu_sq"]
ord = 2
stp = 1e-4
N = 5
krylovdim = 20

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "rotate" => rotate,
)

phi4_pars = Dict(
    "mu_sq" => mu_sq,
    "lam" => lam,
    "kappa" => kappa,
    "K" => K,
    "D" => D,
    "symmetry_tensors" => true
)

# Helper functions (copied from phi4_eigensystem.jl)
function gilt_phi4(A_ju)
    A_py = ju_to_py(A_ju)
    getitem = pyimport("operator").getitem
    result = pycall(py"gilttnr_step", PyObject, A_py, 0.0, gilt_pars)
    A_out = pycall(getitem, PyObject, result, 0)
    A_out, _ = fix_gauge_phi4(A_out; method=:environment)
    return py_to_ju(A_out)
end

function project_to_structure(A::Z2Tensor, template::Z2Tensor)
    new_sects = Dict{NTuple{4, Int64}, Array}()
    for (k, v_temp) in template.sects
        if haskey(A.sects, k)
            v_A = A.sects[k]
            if size(v_A) == size(v_temp)
                new_sects[k] = v_A
            else
                new_v = zeros(eltype(v_A), size(v_temp))
                common_size = min.(size(v_A), size(v_temp))
                ranges = [1:s for s in common_size]
                new_v[ranges...] = v_A[ranges...]
                new_sects[k] = new_v
            end
        else
            new_sects[k] = zeros(eltype(template.sects[k]), size(template.sects[k]))
        end
    end
    return Z2Tensor(new_sects, template.shape, template.qhape, template.dirs)
end

function random_like(A)
    new_sects = Dict{NTuple{4, Int64}, Array}()
    for (k, v) in A.sects
        new_sects[k] = rand(eltype(v), size(v))
    end
    return Z2Tensor(new_sects, A.shape, A.qhape, A.dirs)
end

# Scan loop
println("Scanning number_of_initial_steps...")
println("Step | Lambda 1 | Lambda 2 | Lambda 3")

for steps in 4:6
    # RG Flow
    trajectory_result = phi4_trajectory(phi4_pars, steps, gilt_pars)
    A_crit = trajectory_result["A"][end]
    A_prev = trajectory_result["A"][end-1]
    
    # Check convergence
    A_crit_JU = py_to_ju(A_crit)
    A_prev_JU = py_to_ju(A_prev)
    
    # Need to align them before diff? They might have different gauge.
    # But fix_gauge_phi4 handles gauge fixing.
    A_crit_fixed, _ = fix_gauge_phi4(A_crit; method=:environment)
    A_prev_fixed, _ = fix_gauge_phi4(A_prev; method=:environment)
    
    A_crit_JU = py_to_ju(A_crit_fixed)
    A_prev_JU = py_to_ju(A_prev_fixed)
    
    # Simple norm diff (might fail if shapes differ)
    try
        diff = norm(A_crit_JU - A_prev_JU)
        println("Step $steps: Diff = $diff")
    catch e
        println("Step $steps: Diff = Shape Mismatch")
    end

    # Linearized RG
    function linearized_RG(dA)
        out = NumDifferentiation.df(gilt_phi4, A_crit_JU, dA; stp=stp, order=ord)
        return project_to_structure(out, A_crit_JU)
    end

    # Eigsolve
    x0 = random_like(A_crit_JU)
    vals, vecs, info = eigsolve(linearized_RG, x0, N, :LM; krylovdim=krylovdim, verbosity=0)
    
    l1 = length(vals) >= 1 ? round(real(vals[1]), digits=4) : NaN
    l2 = length(vals) >= 2 ? round(real(vals[2]), digits=4) : NaN
    l3 = length(vals) >= 3 ? round(real(vals[3]), digits=4) : NaN
    
    println("$steps | $l1 | $l2 | $l3")
end
