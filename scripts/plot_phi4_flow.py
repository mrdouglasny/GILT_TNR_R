import matplotlib.pyplot as plt
import numpy as np
import os

def parse_data(filepath):
    steps = []
    x_sigmas = []
    x_epsilons = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Skip header
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            step = int(parts[0])
            # Check for -0.0 which might be x_identity (0)
            # Scaling dimensions usually ordered.
            # File format: Step x_sigma x_epsilon ...
            # Actually line 2 says: 1 -0.0 0.119 ...
            # So col 1 is x_0=0, col 2 is x_sigma, col 3 is x_epsilon
            
            # parts[0] is Step
            # parts[1] is 0.0 (identity)
            # parts[2] is x_sigma
            # parts[3] is x_epsilon
            
            steps.append(step)
            x_sigmas.append(float(parts[2]))
            x_epsilons.append(float(parts[3]))
        except ValueError:
            continue
            
    return steps, x_sigmas, x_epsilons

def plot_exponents(data_file, output_file):
    steps, sigmas, epsilons = parse_data(data_file)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Sigma
    plt.plot(steps, sigmas, 'o-', label=r'$x_\sigma$ (meas)', color='blue')
    plt.axhline(y=0.125, color='blue', linestyle='--', alpha=0.5, label=r'$x_\sigma$ (Ising: 0.125)')
    
    # Plot Epsilon
    plt.plot(steps, epsilons, 's-', label=r'$x_\epsilon$ (meas)', color='red')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label=r'$x_\epsilon$ (Ising: 1.0)')
    
    plt.xlabel('RG Step')
    plt.ylabel('Scaling Dimension')
    plt.title(r'$\phi^4$ Critical Exponents Flow ($\mu^2 \approx 2.7318$)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Zoom in reasonably
    plt.ylim(0, 1.5)
    
    print(f"Saving plot to {output_file}")
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Phi4 RG flow")
    parser.add_argument("input_file", nargs="?", help="Path to data file")
    parser.add_argument("output_file", nargs="?", help="Path to output PNG")
    args = parser.parse_args()

    default_data_path = "/workspaces/rg/ekrgilttrnr/data/phi4_exponents/phi4_exponents_mu2.731815_chi32.dat"
    data_path = args.input_file if args.input_file else default_data_path
    
    default_output_dir = "/workspaces/rg/ekrgilttrnr/docs/figures"
    os.makedirs(default_output_dir, exist_ok=True)
    default_output_path = os.path.join(default_output_dir, "phi4_flow.png")
    output_path = args.output_file if args.output_file else default_output_path
    
    # Ensure directory exists for output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(data_path):
        plot_exponents(data_path, output_path)
    else:
        print(f"Data file not found: {data_path}")
