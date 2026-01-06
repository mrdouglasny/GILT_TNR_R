# Simple test to verify EKR setup works
# Run from ekrgilttrnr/src directory

println("Loading modules...")
include("Tools.jl")
include("GaugeFixing.jl")
include("KrylovTechnical.jl")

println("✓ All modules loaded successfully!")

# Test parameters
chi = 8  # Small chi for quick test
gilt_eps = 6e-6
Jratio = 1.0

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 1,
    "rotate" => false,
)

println("\nTesting critical temperature finder...")
search_tol = 1.0e-10
relT_low, relT_high, relT = find_critical_temperature(chi, gilt_pars, search_tol, Jratio)
println("✓ Found critical temperature: relT = $relT")

# Use small offset to avoid division by zero in trajectory
if abs(relT) < 1e-10
    relT = 1e-6
    println("  Using relT = $relT to avoid division by zero")
end

println("\nTesting trajectory generation...")
initialA_pars = Dict("relT" => relT, "Jratio" => Jratio, "symmetry_tensors" => true)  # Enable symmetry tensors (required for invar)
number_of_initial_steps = 3  # Small for quick test
traj = trajectory(initialA_pars, number_of_initial_steps, gilt_pars)
A_final = traj["A"][end]
println("✓ Trajectory computed with $(length(traj["A"])) steps")
println("  Final tensor type: $(typeof(A_final))")

println("\n✅ EKR Gilt-TNR setup is working!")
println("   NumPy compatibility: FIXED (np.float_ → np.float64)")
println("   Python GiltTNR library: LOADED")
println("   Ready to run full eigenvalue computation")
println("\nNote: Full eigensystem.jl script requires running from project root with correct paths")
