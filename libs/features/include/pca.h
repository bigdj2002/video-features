#pragma once

#include <armadillo>
#include <vector>

class PCA
{
public:
  PCA(unsigned int n_components);

  void fit(const std::vector<std::vector<double>> &X);
  std::vector<std::vector<double>> transform(const arma::mat &X);
  std::vector<std::vector<double>> fit_transform(const std::vector<std::vector<double>> &X);

private:
  unsigned int n_components;
  arma::rowvec mean;
  arma::mat components;

  arma::mat toArmaMat(const std::vector<std::vector<double>> &X_vec);
  std::vector<std::vector<double>> toStdVec(const arma::mat &X_mat);
};
