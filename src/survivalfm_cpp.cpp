#include <Rcpp.h>
#include <RcppEigen.h>
#include <future>

using namespace Eigen;
using namespace std;


Eigen::VectorXd cumulative_sum(const Eigen::VectorXd& vec) {
  Eigen::VectorXd cumsum(vec.size());
  double sum = 0.0;
  for (int i = 0; i < vec.size(); ++i) {
    sum += vec[i];
    cumsum[i] = sum;
  }
  return cumsum;
}


// [[Rcpp::export]]
Eigen::VectorXd predict_cpp(
    Eigen::MatrixXd X,
    Eigen::VectorXd beta,
    Eigen::MatrixXd P,
    Eigen::MatrixXd XP,
    bool fit_interactions
) {
  Eigen::VectorXd lp = Eigen::VectorXd::Zero(X.rows());
  if (fit_interactions == true) {
    auto linear_terms_future = std::async(std::launch::async, [&](){return X * beta;});
    auto interactions_future = std::async(std::launch::async, [&](){return (0.5 * ((XP.array().square().matrix().rowwise().sum()) - ((X.array().square().matrix() * P.array().square().matrix()).rowwise().sum())));});
    Eigen::VectorXd linear_terms = linear_terms_future.get();
    Eigen::VectorXd interactions = interactions_future.get();
    lp = linear_terms + interactions;
  } else {
    lp = X * beta;
  }
  return lp;
}

// [[Rcpp::export]]
double negative_log_likelihood(
    const Eigen::VectorXd& linear_predictors,
    const Eigen::VectorXi& status,
    const Eigen::VectorXd& time
) {
  Eigen::VectorXd exp_lp = linear_predictors.array().exp();
  Eigen::VectorXd cumsum = cumulative_sum(exp_lp);
  double log_likelihood = 0.0;
  double epsilon = 1e-8;
  for (int i = 0; i < time.size(); ++i) {
    if (status[i] == 1) {
      int tie_end = i;
      while (tie_end + 1 < time.size() && time[tie_end + 1] == time[i] && status[tie_end + 1] == 1) {
        ++tie_end;
      }
      log_likelihood += linear_predictors[i] - std::log(max(cumsum[tie_end], epsilon));
    }
  }
  return -log_likelihood;
}

// [[Rcpp::export]]
double regularized_negative_log_likelihood(
    const Eigen::VectorXi status,
    const Eigen::VectorXd time,
    const Eigen::VectorXd lp,
    const Eigen::VectorXd cumsum_exp_lp,
    const Eigen::MatrixXd P,
    const Eigen::VectorXd beta,
    double lambda1,
    double lambda2,
    bool fit_interactions
) {

  double log_likelihood = 0.0;
  double reg_log_likelihood = 0.0;
  double epsilon = 1e-8;
  for (int i = 0; i < time.size(); ++i) {
    if (status[i] == 1) {
      int tie_end = i;
      while (tie_end + 1 < time.size() && time[tie_end + 1] == time[i] && status[tie_end + 1] == 1) {
        ++tie_end;
      }
      log_likelihood += lp[i] - std::log(max(cumsum_exp_lp[tie_end], epsilon));
    }
  }

  double multiplier = (2.0 / lp.size());
  reg_log_likelihood = multiplier * -log_likelihood  + 0.5 * lambda1 * beta.array().square().sum();
  if (fit_interactions == true) {
    reg_log_likelihood  = reg_log_likelihood  +  0.5 * lambda2 * P.array().square().sum();
  }

  return reg_log_likelihood;
}


// [[Rcpp::export]]
Rcpp::List shared_computations(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& beta,
    const Eigen::MatrixXd& P,
    bool fit_interactions
) {
  Eigen::MatrixXd XP = X * P;
  Eigen::VectorXd lp = predict_cpp(X, beta, P, XP, fit_interactions);
  Eigen::VectorXd exp_lp = lp.array().exp();
  Eigen::VectorXd cumsum_exp_lp = cumulative_sum(exp_lp).array();

  return Rcpp::List::create(
    Rcpp::Named("XP") = XP,
    Rcpp::Named("lp") = lp,
    Rcpp::Named("exp_lp") = exp_lp,
    Rcpp::Named("cumsum_exp_lp") = cumsum_exp_lp
  );

}

void compute_grad_beta(
    Eigen::VectorXd& grad_beta,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& beta,
    const Eigen::VectorXd& exp_lp,
    const Eigen::VectorXi& status,
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& cumsum_exp_lp,
    double lambda1
) {

  Eigen::VectorXd cumsum_t1_beta = Eigen::VectorXd::Zero(X.cols());
  Eigen::VectorXd partial_beta = Eigen::VectorXd::Zero(X.cols());
  for (int i = 0; i < time.size(); ++i) {
    partial_beta = X.row(i);
    cumsum_t1_beta += (exp_lp(i) * partial_beta);
    if (status[i] == 1) {
      int tie_end = i;
      while (tie_end + 1 < time.size() && time[tie_end + 1] == time[i] && status[tie_end + 1] == 1) {
        ++tie_end;
      }
      grad_beta += (partial_beta - cumsum_t1_beta / cumsum_exp_lp[tie_end]);
    }
  }
  double multiplier = (2.0 / X.rows());
  grad_beta = -multiplier * grad_beta + lambda1 * beta;
}


void compute_grad_P(
    Eigen::MatrixXd& grad_P,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& P,
    const Eigen::MatrixXd& XP,
    const Eigen::VectorXd& exp_lp,
    const Eigen::VectorXi& status,
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& cumsum_exp_lp,
    double lambda2
) {

  Eigen::MatrixXd cumsum_t1_P = Eigen::MatrixXd::Zero(P.rows(), P.cols());
  Eigen::MatrixXd partial_P = Eigen::MatrixXd::Zero(P.rows(), P.cols());

  for (int i = 0; i < time.size(); ++i) {
    for (int col = 0; col < X.cols(); col++) {
      for (int f = 0; f < P.cols(); f++) {
        partial_P(col, f) = X(i, col) * XP(i, f) - P(col,f) * X(i, col) * X(i, col);
      }
    }

    cumsum_t1_P +=  exp_lp(i) * partial_P;

    if (status[i] == 1) {
      int tie_end = i;
      while (tie_end + 1 < time.size() && time[tie_end + 1] == time[i] && status[tie_end + 1] == 1) {
        ++tie_end;
      }
      grad_P += (partial_P - cumsum_t1_P / cumsum_exp_lp[tie_end]);
    }
  }
  double multiplier = (2.0 / X.rows());
  grad_P = -multiplier * grad_P + lambda2 * P;
}


// [[Rcpp::export]]
Rcpp::List compute_gradients(
    Eigen::MatrixXd X,
    Eigen::VectorXi status,
    Eigen::VectorXd time,
    Eigen::MatrixXd P,
    Eigen::VectorXd beta,
    Eigen::MatrixXd& XP,
    Eigen::VectorXd& exp_lp,
    Eigen::VectorXd& cumsum_exp_lp,
    double lambda1,
    double lambda2,
    bool fit_interactions
) {

  Eigen::VectorXd grad_beta = Eigen::VectorXd::Zero(X.cols());
  Eigen::MatrixXd grad_P = Eigen::MatrixXd::Zero(P.rows(), P.cols());

  if (fit_interactions) {
    // Asynchronous computation of gradients
    auto future_grad_beta = std::async(
      std::launch::async,
      compute_grad_beta,
      std::ref(grad_beta), std::ref(X), std::ref(beta), std::ref(exp_lp), std::ref(status), std::ref(time), std::ref(cumsum_exp_lp), lambda1);
    auto future_grad_P = std::async(
      std::launch::async,
      compute_grad_P,
      std::ref(grad_P), std::ref(X), std::ref(P), std::ref(XP), std::ref(exp_lp), std::ref(status), std::ref(time), std::ref(cumsum_exp_lp), lambda2);

    future_grad_beta.get();
    future_grad_P.get();
  } else {
    compute_grad_beta(grad_beta, X, beta, exp_lp, status, time, cumsum_exp_lp, lambda1);
  }
  return Rcpp::List::create(Rcpp::Named("grad_beta") = grad_beta,
                            Rcpp::Named("grad_P") = grad_P);
}

