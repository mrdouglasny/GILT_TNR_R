
using Serialization
using LinearAlgebra
using KrylovKit
using TOML
using PyCall

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
base_mu_sq = config["mu_sq"]

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "rotate" => rotate,
)

# Helper functions
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

println("Scanning mu_sq around $base_mu_sq")
println("mu_sq | Step 8 Lambda 1 | Step 9 Lambda 1")

# Scan range: +/- 5%
shifts = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]

for shift in shifts
    current_mu_sq = base_mu_sq * (1.0 + shift)
    
    phi4_pars = Dict(
        "mu_sq" => current_mu_sq,
        "lam" => lam,
        "kappa" => kappa,
        "K" => K,
        "D" => D,
        "symmetry_tensors" => true
    )

    # Step 8
    traj_8 = phi4_trajectory(phi4_pars, 8, gilt_pars)
    A_py = traj_8["A"][end]
    A_fixed_py, _ = fix_gauge_phi4(A_py; method=:environment)
    A_8 = py_to_ju(A_fixed_py)
    
    function lin_map_8(v)
        out = NumDifferentiation.df(gilt_phi4, A_8, v; stp=1e-4, order=2)
        return project_to_structure(out, A_8)
    end
    
    v0 = random_like(A_8)
    vals_8, _ = eigsolve(lin_map_8, v0, 1, :LM, krylovdim=20)
    lam_8 = real(vals_8[1])

    # Step 9
    # Continue trajectory from A_8? No, phi4_trajectory runs from scratch.
    # But we can just run one more step from A_8.
    # Actually, phi4_trajectory returns the tensor AT that step.
    # So let's just run gilt_phi4 on A_8 to get A_9.
    
    A_9 = gilt_phi4(A_8)
    
    function lin_map_9(v)
        out = NumDifferentiation.df(gilt_phi4, A_9, v; stp=1e-4, order=2)
        return project_to_structure(out, A_9)
    end
    
    v0_9 = random_like(A_9)
    vals_9, _ = eigsolve(lin_map_9, v0_9, 1, :LM, krylovdim=20)
    lam_9 = real(vals_9[1])

    println("$current_mu_sq | $lam_8 | $lam_9")
end
