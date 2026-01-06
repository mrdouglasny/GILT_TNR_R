# MODULE: phi4_critical_mu_sq.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_critical_mu_sq.jl [options]
# INPUTS: None (parameters from command line)
# OUTPUTS: ekrgilttrnr/critical_temperatures/phi4_mu_sq_*.data (serialized critical μ²)
# DESCRIPTION: Binary search to find critical μ² for 2D φ⁴ model
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312),
#           ekrgilttrnr/scripts/critical_temperature.jl adapted for φ⁴
# NEW IN THIS SCRIPT: Binary search over μ² (instead of temperature) to locate
#                     φ⁴ critical point where RG flow changes behavior.
#
# RUNTIME: ~20-30 min at χ=30. Depends on binary search precision.

################################################
# section: EXPERIMENT INIT
################################################

using ArgParse
using Serialization

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--chi"
    help = "Maximum bond dimension for Gilt-TNR"
    arg_type = Int64
    default = 30
    "--gilt_eps"
    help = "Gilt truncation threshold"
    arg_type = Float64
    default = 6e-6
    "--cg_eps"
    help = "TRG truncation threshold"
    arg_type = Float64
    default = 1e-10
    "--rotate"
    help = "Whether to rotate 90° after each RG step"
    arg_type = Bool
    default = false
    "--lam"
    help = "Quartic coupling λ"
    arg_type = Float64
    default = 1.0
    "--kappa"
    help = "Kinetic coupling κ"
    arg_type = Float64
    default = 1.0
    "--K"
    help = "Number of quadrature points"
    arg_type = Int64
    default = 32
    "--D"
    help = "Initial bond dimension for φ⁴ tensor"
    arg_type = Int64
    default = 16
    "--mu_sq_low"
    help = "Lower bound for μ² search (broken phase)"
    arg_type = Float64
    default = -0.2
    "--mu_sq_high"
    help = "Upper bound for μ² search (symmetric phase)"
    arg_type = Float64
    default = 0.0
    "--search_tol"
    help = "Tolerance for binary search"
    arg_type = Float64
    default = 1e-8
    "--verbosity"
    help = "Verbosity level"
    arg_type = Int64
    default = 2
end

pars = parse_args(settings; as_symbols = true)
for (key, value) in pars
    @eval $key = $value
end

# Load tools
include("../src/Tools.jl")
include("../src/Phi4Tools.jl")

################################################
# section: MAIN
################################################

@info "=" ^ 60
@info "φ⁴ Critical Point Search"
@info "=" ^ 60

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => verbosity,
    "rotate" => rotate,
)

@info "Gilt-TNR parameters:"
@info "  χ = $chi"
@info "  gilt_eps = $gilt_eps"
@info "  cg_eps = $cg_eps"
@info "  rotate = $rotate"

@info "\nφ⁴ model parameters:"
@info "  λ = $lam"
@info "  κ = $kappa"
@info "  K = $K (quadrature points)"
@info "  D = $D (initial bond dim)"

@info "\nSearch parameters:"
@info "  μ² range: [$mu_sq_low, $mu_sq_high]"
@info "  tolerance: $search_tol"

# Perform the search
@time mu_sq_low_final, mu_sq_high_final, mu_sq_critical = find_critical_mu_sq(
    lam, kappa, K, D, gilt_pars;
    mu_sq_low = mu_sq_low,
    mu_sq_high = mu_sq_high,
    search_tol = search_tol,
    verbose = (verbosity > 0)
)

@info "\n" * "=" ^ 60
@info "RESULT: Critical μ² ≈ $mu_sq_critical"
@info "Range: [$mu_sq_low_final, $mu_sq_high_final]"
@info "=" ^ 60

# Save results
mkpath("critical_temperatures")

result = Dict(
    "mu_sq_low" => mu_sq_low_final,
    "mu_sq_high" => mu_sq_high_final,
    "mu_sq_critical" => mu_sq_critical,
    "lam" => lam,
    "kappa" => kappa,
    "K" => K,
    "D" => D,
    "gilt_pars" => gilt_pars,
)

filename = "critical_temperatures/phi4_" * gilt_pars_identifier(gilt_pars) *
           "_lam=$(lam)_kappa=$(kappa)_K=$(K)_D=$(D)_tol=$(search_tol).data"

serialize(filename, result)
@info "\nResults saved to: $filename"
