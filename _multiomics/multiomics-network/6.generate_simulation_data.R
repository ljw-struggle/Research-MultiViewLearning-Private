# Reference: https://github.com/wenpingd/JRmGRN
library(MASS)

four_genes_theta_cov <- function(file_header, p = 4, sample_n = 1000, mu_zero = FALSE) {
  mat <- matrix(0, 4, 4)
  mat[1, 2] <- ifelse(runif(1, min = 0, max = 100) > 50, runif(1, min = 0.25, max = 1) * sample(c(1, -1), 1), 0)
  mat[2, 1] <- mat[1, 2]
  mat[1, 3] <- ifelse(runif(1, min = 0, max = 100) > 50, runif(1, min = 0.25, max = 1) * sample(c(1, -1), 1), 0)
  mat[3, 1] <- mat[1, 3]
  mat[1, 4] <- ifelse(runif(1, min = 0, max = 100) > 50, runif(1, min = 0.25, max = 1) * sample(c(1, -1), 1), 0)
  mat[4, 1] <- mat[1, 4]
  mat[2, 3] <- ifelse(runif(1, min = 0, max = 100) > 50, runif(1, min = 0.25, max = 1) * sample(c(1, -1), 1), 0)
  mat[3, 2] <- mat[2, 3]
  mat[2, 4] <- ifelse(runif(1, min = 0, max = 100) > 50, runif(1, min = 0.25, max = 1) * sample(c(1, -1), 1), 0)
  mat[4, 2] <- mat[2, 4]
  mat[3, 4] <- ifelse(runif(1, min = 0, max = 100) > 50, runif(1, min = 0.25, max = 1) * sample(c(1, -1), 1), 0)
  mat[4, 3] <- mat[3, 4]
  theta <- mat

  ee <- min(eigen(theta, only.values = TRUE)$values)
  diag(theta) <- ifelse(ee < 0, -ee + 0.1, 0.1)
  cov <- ginv(theta)

  mu_vector <- ifelse(rep(mu_zero, p), rep(0, p), runif(p, 10, 10000))
  sim_data <- mvrnorm(sample_n, mu = mu_vector, Sigma = cov, tol = 1e-6)
  sim_data <- abs(sim_data)

  write.table(theta, file = paste(file_header, "theta.csv", sep = ""), quote = FALSE, sep = ",", col.names = FALSE, row.names = FALSE)
  write.table(cov, file = paste(file_header, "cov.csv", sep = ""), quote = FALSE, sep = ",", col.names = FALSE, row.names = FALSE)
  write.table(sim_data, file = paste(file_header, "sim_data.csv", sep = ""), quote = FALSE, sep = ",")
}


more_genes_theta_cov <- function(file_header, file_header_1, file_header_2, file_header_3, p = 100, sparsity = 0.8, sample_n = 1000, mu_zero = FALSE) {
  sparse.base <- rbinom(p * p, 1, 1 - sparsity) * sample(c(-1, 1), p * p, replace = TRUE) * runif(p * p, 0.25, 0.75)
  theta <- matrix(data = sparse.base, p, p)
  theta[lower.tri(theta, diag = FALSE)] <- 0
  theta <- theta + t(theta)
  theta <- ifelse(abs(theta) < 1e-5, 0, theta)
  diag(theta) <- 0

  ee <- min(eigen(theta, only.values = TRUE)$values)
  diag(theta) <- ifelse(ee < 0, -ee + 0.1, 0.1)
  cov <- ginv(theta)

  mu_vector <- ifelse(rep(mu_zero, p), rep(0, p), runif(p, 10, 10000))
  sim_data_1 <- mvrnorm(sample_n, mu = mu_vector, Sigma = cov, tol = 1e-6)
  sim_data_1 <- abs(sim_data_1)
  sim_data_2 <- mvrnorm(sample_n, mu = mu_vector, Sigma = cov, tol = 1e-6)
  sim_data_2 <- abs(sim_data_2)
  sim_data_3 <- mvrnorm(sample_n, mu = mu_vector, Sigma = cov, tol = 1e-6)
  sim_data_3 <- abs(sim_data_3)

  write.table(theta, file = paste(file_header, "theta.csv", sep = ""), quote = FALSE, sep = ",", col.names = FALSE, row.names = FALSE)
  write.table(cov, file = paste(file_header, "cov.csv", sep = ""), quote = FALSE, sep = ",", col.names = FALSE, row.names = FALSE)
  write.table(sim_data_1, file = paste0(file_header_1, "_sim_data.csv", sep = ""), quote = FALSE, sep = ",")
  write.table(sim_data_2, file = paste0(file_header_2, "_sim_data.csv", sep = ""), quote = FALSE, sep = ",")
  write.table(sim_data_3, file = paste0(file_header_3, "_sim_data.csv", sep = ""), quote = FALSE, sep = ",")
}


more_genes_theta_cov_with_diff <- function(file_header, theta, p = 100, diff_rate = 0.1, sample_n = 1000, mu_zero = FALSE) {
  # there will be 10% of the elements in theta that will be changed when diff_rate = 0.1
  index <- c(1:(p * p))
  index <- sample(index, p * p * diff_rate, replace = FALSE)
  theta[index] <- sample(c(-1, 1), length(index), replace = TRUE) * runif(length(index), 0.25, 0.75)
  theta[lower.tri(theta, diag = FALSE)] <- 0
  theta <- theta + t(theta)
  theta <- ifelse(abs(theta) < 1e-5, 0, theta)
  diag(theta) <- 0

  ee <- min(eigen(theta, only.values = TRUE)$values)
  diag(theta) <- ifelse(ee < 0, -ee + 0.1, 0.1)
  cov <- ginv(theta)

  mu_vector <- ifelse(rep(mu_zero, p), rep(0, p), runif(p, 10, 10000))
  sim_data <- mvrnorm(sample_n, mu = mu_vector, Sigma = cov, tol = 1e-6)
  sim_data <- abs(sim_data)

  write.table(theta, file = paste(file_header, "theta.csv", sep = ""), quote = FALSE, sep = ",", col.names = FALSE, row.names = FALSE)
  write.table(cov, file = paste(file_header, "cov.csv", sep = ""), quote = FALSE, sep = ",", col.names = FALSE, row.names = FALSE)
  write.table(sim_data, file = paste(file_header, "sim_data.csv", sep = ""), quote = FALSE, sep = ",")

  result <- list(theta = theta, sim_data = sim_data)
  return(result)
}

# generate data for simulation 100 genes 1000 cell 0.8 sparsity
# theta.csv: the true network
# cov.csv: the covariance matrix
# sim_data.csv: the simulated data
genes <- 100
cells <- 1000
output_dir <- paste("./simulation_data/", "simulation_gene_", genes, "_cell_", cells, "_sr_0.8/", sep = "")
dir.create(output_dir, recursive = TRUE)

num_network <- 10
for (i in c(1:num_network)){
  cat("i=", i, "\n")
  output_dir_network <- paste(output_dir, "network_", i, "/", sep = "")
  dir.create(output_dir_network)
  file_header <- output_dir_network
  file_header_1 <- paste(output_dir_network, "1", sep = "")
  file_header_2 <- paste(output_dir_network, "2", sep = "")
  file_header_3 <- paste(output_dir_network, "3", sep = "")
  more_genes_theta_cov(file_header, file_header_1, file_header_2, file_header_3, p = genes, sparsity = 0.8, sample_n = cells, mu_zero = FALSE)
}


# run the following command in the terminal
# $ nohup R -f generate_simulation_data.R > generate_simulation_data.log 2>&1 &
# $ ps -ef | grep R | grep -v grep | awk '{print $2}' | xargs kill -9