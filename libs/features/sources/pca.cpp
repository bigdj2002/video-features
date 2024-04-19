#include "pca.h"

PCA::PCA(unsigned int n_components_) : n_components(n_components_) {}

void PCA::fit(const std::vector<std::vector<double>> &X_vec)
{
  arma::mat X(X_vec.size(), X_vec[0].size());
  for (size_t i = 0; i < X_vec.size(); ++i)
  {
    for (size_t j = 0; j < X_vec[i].size(); ++j)
    {
      X(i, j) = X_vec[i][j];
    }
  }
  mean = arma::mean(X, 0);
  arma::mat X_centered = X.each_row() - mean;
  arma::mat covariance_matrix = (X_centered.t() * X_centered) / double(X.n_rows - 1);
  arma::vec eigenvalues;
  arma::mat eigenvectors;
  arma::eig_sym(eigenvalues, eigenvectors, covariance_matrix);
  arma::uvec indices = arma::sort_index(eigenvalues, "descend");
  components = eigenvectors.cols(indices.head(n_components));
}

std::vector<std::vector<double>> PCA::transform(const arma::mat &X)
{
  arma::mat X_centered = X.each_row() - mean;
  arma::mat X_transformed = X_centered * components;
  std::vector<std::vector<double>> result(X_transformed.n_rows, std::vector<double>(X_transformed.n_cols));
  for (size_t i = 0; i < X_transformed.n_rows; ++i)
  {
    for (size_t j = 0; j < X_transformed.n_cols; ++j)
    {
      result[i][j] = X_transformed(i, j);
    }
  }
  return result;
}

std::vector<std::vector<double>> PCA::fit_transform(const std::vector<std::vector<double>> &X_vec)
{
  arma::mat X(X_vec.size(), X_vec[0].size());
  for (size_t i = 0; i < X_vec.size(); ++i)
  {
    for (size_t j = 0; j < X_vec[i].size(); ++j)
    {
      X(i, j) = X_vec[i][j];
    }
  }

  fit(X_vec);

  return transform(X);
}