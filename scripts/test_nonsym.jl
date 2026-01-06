# Test without Z2 symmetry enforcement
using PyCall
using LinearAlgebra

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "../src/GiltTNR"))
include(joinpath(@__DIR__, "../src/Tools.jl"))
include(joinpath(@__DIR__, "../src/Phi4Tools.jl"))

const chi = 50
const gilt_pars = Dict(
    "gilt_eps" => 6e-6,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "rotate" => false,
)

function test_trajectory(mu_sq; use_symmetry=true)
    label = use_symmetry ? "Z2-symmetric" : "non-symmetric"
    println("\n=== Testing $label tensor at μ² = $mu_sq ===\n")

    phi4_pars = Dict(
        "mu_sq" => mu_sq,
        "lam" => 1.0,
        "kappa" => 0.3,
        "K" => 32,
        "D" => 16,
        "symmetry_tensors" => use_symmetry
    )

    A = initial_tensor_phi4(phi4_pars)
    println("Step 0: shape = $(A.shape)")

    getitem = pyimport("operator").getitem
    for step in 1:20
        result = pycall(py"gilttnr_step", PyObject, A, 0.0, gilt_pars)
        A = pycall(getitem, PyObject, result, 0)

        # Get spectrum
        spectrum = py"get_A_spectrum_phi4"(A)
        n_spec = length(spectrum)

        ev2 = n_spec >= 2 ? spectrum[2] : 0.0
        ev3 = n_spec >= 3 ? spectrum[3] : 0.0

        # Get shape - handle differently for symmetric vs non-symmetric
        shape = A.shape
        if isa(shape, Tuple)
            max_dim = maximum(shape)
        else
            # Z2 tensor has nested structure
            shape_flat = vcat([collect(s) for s in shape]...)
            max_dim = maximum(shape_flat)
        end

        println("Step $step: dim=$max_dim, λ₂=$(round(ev2, digits=5)), λ₃=$(round(ev3, digits=5))")

        if max_dim <= 1
            println("  ** COLLAPSED **")
            break
        end
    end
end

# Test both at critical point
test_trajectory(-1.325; use_symmetry=true)
test_trajectory(-1.325; use_symmetry=false)
