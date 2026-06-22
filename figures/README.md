# Figure-generation code

Each Python script creates one main-paper figure or one supplementary figure.

The scripts read the simulation summaries in `results/` and write PDF outputs to
this directory or its `supplementary/` subdirectory. Generated PDFs are ignored by Git.

The code reproduces the simulation design, real-data analysis pipeline, and qualitative conclusions of the paper. Exact numerical equality across runs is not expected because the procedures involve random data generation and injected Gaussian noise.
