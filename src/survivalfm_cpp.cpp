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
    Eigen::VectorXd w,
    Eigen::MatrixXd V,
    Eigen::MatrixXd XV,
    bool fit_interactions
) {
  Eigen::VectorXd lp = Eigen::VectorXd::Zero(X.rows());
  if (fit_interactions == true) {
    auto linear_terms_future = std::async(std::launch::async, [&](){return X * w;});
    auto interactions_future = std::async(std::launch::async, [&](){return (0.5 * ((XV.array().square().matrix().rowwise().sum()) - ((X.array().square().matrix() * V.array().square().matrix()).rowwise().sum())));});
    Eigen::VectorXd linear_terms = linear_terms_future.get();
    Eigen::VectorXd interactions = interactions_future.get();
    lp = linear_terms + interactions;
  } else {
    lp = X * w;
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
    const Eigen::MatrixXd V,
    const Eigen::VectorXd w,
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
  reg_log_likelihood = multiplier * -log_likelihood  + 0.5 * lambda1 * w.array().square().sum();
  if (fit_interactions == true) {
    reg_log_likelihood  = reg_log_likelihood  +  0.5 * lambda2 * V.array().square().sum();
  }

  return reg_log_likelihood;
}


// [[Rcpp::export]]
Rcpp::List shared_computations(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& w,
    const Eigen::MatrixXd& V,
    bool fit_interactions
) {
  Eigen::MatrixXd XV = X * V;
  Eigen::VectorXd lp = predict_cpp(X, w, V, XV, fit_interactions);
  Eigen::VectorXd exp_lp = lp.array().exp();
  Eigen::VectorXd cumsum_exp_lp = cumulative_sum(exp_lp).array();

  return Rcpp::List::create(
    Rcpp::Named("XV") = XV,
    Rcpp::Named("lp") = lp,
    Rcpp::Named("exp_lp") = exp_lp,
    Rcpp::Named("cumsum_exp_lp") = cumsum_exp_lp
  );

}

void compute_grad_w(
    Eigen::VectorXd& grad_w,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& w,
    const Eigen::VectorXd& exp_lp,
    const Eigen::VectorXi& status,
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& cumsum_exp_lp,
    double lambda1
) {

  Eigen::VectorXd cumsum_t1_w = Eigen::VectorXd::Zero(X.cols());
  Eigen::VectorXd partial_w = Eigen::VectorXd::Zero(X.cols());
  for (int i = 0; i < time.size(); ++i) {
    partial_w = X.row(i);
    cumsum_t1_w += (exp_lp(i) * partial_w);
    if (status[i] == 1) {
      int tie_end = i;
      while (tie_end + 1 < time.size() && time[tie_end + 1] == time[i] && status[tie_end + 1] == 1) {
        ++tie_end;
      }
      grad_w += (partial_w - cumsum_t1_w / cumsum_exp_lp[tie_end]);
    }
  }
  double multiplier = (2.0 / X.rows());
  grad_w = -multiplier * grad_w + lambda1 * w;
}


void compute_grad_V(
    Eigen::MatrixXd& grad_V,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& XV,
    const Eigen::VectorXd& exp_lp,
    const Eigen::VectorXi& status,
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& cumsum_exp_lp,
    double lambda2
) {

  Eigen::MatrixXd cumsum_t1_V = Eigen::MatrixXd::Zero(V.rows(), V.cols());
  Eigen::MatrixXd partial_V = Eigen::MatrixXd::Zero(V.rows(), V.cols());

  for (int i = 0; i < time.size(); ++i) {
    for (int col = 0; col < X.cols(); col++) {
      for (int f = 0; f < V.cols(); f++) {
        partial_V(col, f) = X(i, col) * XV(i, f) - V(col,f) * X(i, col) * X(i, col);
      }
    }

    cumsum_t1_V +=  exp_lp(i) * partial_V;

    if (status[i] == 1) {
      int tie_end = i;
      while (tie_end + 1 < time.size() && time[tie_end + 1] == time[i] && status[tie_end + 1] == 1) {
        ++tie_end;
      }
      grad_V += (partial_V - cumsum_t1_V / cumsum_exp_lp[tie_end]);
    }
  }
  double multiplier = (2.0 / X.rows());
  grad_V = -multiplier * grad_V + lambda2 * V;
}


// [[Rcpp::export]]
Rcpp::List compute_gradients(
    Eigen::MatrixXd X,
    Eigen::VectorXi status,
    Eigen::VectorXd time,
    Eigen::MatrixXd V,
    Eigen::VectorXd w,
    Eigen::MatrixXd& XV,
    Eigen::VectorXd& exp_lp,
    Eigen::VectorXd& cumsum_exp_lp,
    double lambda1,
    double lambda2,
    bool fit_interactions
) {

  Eigen::VectorXd grad_w = Eigen::VectorXd::Zero(X.cols());
  Eigen::MatrixXd grad_V = Eigen::MatrixXd::Zero(V.rows(), V.cols());

  if (fit_interactions) {
    // Asynchronous computation of gradients
    auto future_grad_w = std::async(
      std::launch::async,
      compute_grad_w,
      std::ref(grad_w), std::ref(X), std::ref(w), std::ref(exp_lp), std::ref(status), std::ref(time), std::ref(cumsum_exp_lp), lambda1);
    auto future_grad_V = std::async(
      std::launch::async,
      compute_grad_V,
      std::ref(grad_V), std::ref(X), std::ref(V), std::ref(XV), std::ref(exp_lp), std::ref(status), std::ref(time), std::ref(cumsum_exp_lp), lambda2);

    future_grad_w.get();
    future_grad_V.get();
  } else {
    compute_grad_w(grad_w, X, w, exp_lp, status, time, cumsum_exp_lp, lambda1);
  }
  return Rcpp::List::create(Rcpp::Named("grad_w") = grad_w,
                            Rcpp::Named("grad_V") = grad_V);
}

