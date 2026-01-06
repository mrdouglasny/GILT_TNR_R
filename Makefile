# Makefile for Phi4 Gilt-TNR Analysis

JULIA := julia
PYTHON := python3
LATEX := pdflatex

# Parameters
MU_SQ := 2.731815
CHI := 32
STEPS := 50

# Directories
DATA_DIR := data/phi4_exponents
DOCS_DIR := docs
FIGS_DIR := $(DOCS_DIR)/figures
SCRIPTS_DIR := scripts

# Files
DATA_FILE := $(DATA_DIR)/phi4_exponents_mu$(MU_SQ)_chi$(CHI).dat
PLOT_FILE := $(FIGS_DIR)/phi4_flow.png
TEX_FILE := $(DOCS_DIR)/phi4_results.tex
PDF_FILE := $(DOCS_DIR)/phi4_results.pdf

.PHONY: all run plot pdf clean help

help:
	@echo "Phi4 Gilt-TNR Analysis Makefile"
	@echo "Targets:"
	@echo "  all   : Run simulation, plotting, and report generation"
	@echo "  run   : Run the Julia Gilt-TNR recursion (may take time)"
	@echo "  plot  : Generate analysis figures"
	@echo "  pdf   : Compile LaTeX report"
	@echo "  clean : Remove generated files"

all: run plot pdf

# 1. Run Simulation
run: $(DATA_FILE)

$(DATA_FILE): $(SCRIPTS_DIR)/phi4_exponents.jl
	@mkdir -p $(DATA_DIR)
	@echo "Running Gilt-TNR simulation (mu^2=$(MU_SQ), chi=$(CHI))..."
	$(JULIA) --project=. $(SCRIPTS_DIR)/phi4_exponents.jl --mu_sq $(MU_SQ) --chi $(CHI) --steps $(STEPS) --output $(DATA_FILE)

# 2. Plot Results
plot: $(PLOT_FILE)

$(PLOT_FILE): $(DATA_FILE) $(SCRIPTS_DIR)/plot_phi4_flow.py
	@mkdir -p $(FIGS_DIR)
	@echo "Generating plot..."
	$(PYTHON) $(SCRIPTS_DIR)/plot_phi4_flow.py $(DATA_FILE) $(PLOT_FILE)

# 3. Compile PDF
pdf: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE) $(PLOT_FILE)
	@echo "Compiling LaTeX report..."
	$(LATEX) -interaction=nonstopmode -output-directory=$(DOCS_DIR) $(TEX_FILE)
	@# Run twice for references
	$(LATEX) -interaction=nonstopmode -output-directory=$(DOCS_DIR) $(TEX_FILE)

clean:
	@echo "Cleaning up..."
	rm -f $(DOCS_DIR)/*.pdf $(DOCS_DIR)/*.aux $(DOCS_DIR)/*.log $(DOCS_DIR)/*.out
	rm -f $(FIGS_DIR)/phi4_flow.png
	# Note: Not deleting data file by default as it is expensive to recompute
	# rm -f $(DATA_FILE)
