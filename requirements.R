# R packages required by figures/Figure7_TSM_positions.R
#
# Run in R:
# source("requirements.R")

required_packages <- c(
  "ggplot2",
  "patchwork"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}
