# *survivalFM*

## Overview

*survivalFM* is an R package designed for efficient modelling of linear and all potential pairwise interaction terms among input predictors in proportional hazards survival models. 

*survivalFM* relies on learning a low-rank factorized representation of the interaction terms, hence overcoming the computational and statistical limitations of directly fitting these terms in the presence of many input variables. The factorization of the interaction parameters, together with an efficient quasi-Newton optimization algorithm, facilitates a systematic exploration of all potential interaction effects across covariates in multivariable time-to-event prediction models involving many predictors.  The resulting model is fully interpretable, providing  access to both individual feature coefficients and those of the approximated interaction terms. 


## Installation


This package can be installed in R with the following command (installation may take a few minutes):

```r
require(devtools)
devtools::install_github("https://github.com/aalto-ics-kepaco/survivalfm")
```

The software has been tested with R version 4.3.1.

## Citation

*survivalFM* is described in the following manuscript:

Heli Julkunen and Juho Rousu. "Machine learning for comprehensive interaction modelling improves disease risk prediction in UK Biobank" (2025).

## Usage example

The following example will demonstrate the usage of *survivalFM* on a small example breast cancer survival dataset. This demo is expected to run within a few seconds.

Running the example requires installation of R packages `tidyverse`, `survival`, `doParallel` and `parallel` (for parallel execution, optional) and `pheatmap` (for visualization). 


### Preprocessing example dataset 

This example uses the publicly available `gbsg` breast cancer survival dataset from the `survival` package. The gbsg data set contains patient records from a 1984-1989 trial conducted by the German Breast Cancer Study Group (GBSG) of 720 patients with node positive breast cancer; it includes 686 patients with complete data on the prognostic variables.


```r
# Load required libraries
library(tidyverse)
library(survival)
library(pheatmap)

### Preparing data ###

# Example dataset
df <- survival::gbsg

# Input covariates
X <- df %>% dplyr::select(age, meno, size, grade, nodes, pgr, er, hormon)

# Time-to-event outcome variable
y <- survival::Surv(time = df$rfstime, event = df$status)

# Split into train and test set
set.seed(123)
training_samples <- X %>% dplyr::sample_frac(0.7) %>% row_number()

X_train <- X[training_samples, ]
y_train <- y[training_samples]
X_test <- X[-training_samples, ]
y_test <- y[-training_samples]

# Scale both train and test sets
X_train_scaled <- X_train %>% scale()

X_test_scaled <- X_test %>%
  scale(center = attr(X_train_scaled, "scaled:center"), scale = attr(X_train_scaled, "scaled:scale"))

```
### Training survivalFM model
 

In the example below, we will use `val.survivalfm()` function, which automatically optimizes the regularization parameters `lambda1` (linear effects) and `lambda2` (factorized interaction parameters) using a validation set taken from the training data. Alternatively, one can use `cv.survivalfm()`, which automatically optimizes the regularization parameters using cross-validation. User only needs to specify the input parameter `rank`, which is the rank defining the dimensionality of the factorization for the interaction parameters. See also the function documentation `?val.survivalfm` for further information.

It is recommended to use multiple cores, if available, to parallelize the optimization process. This can be done by registering the parallel backend using e.g. the `parallel`package, as demonstrated in the example below. 


Using a validation set approach for optimizing regularization parameters: 
```r
library(doParallel)
library(parallel)
library(survivalfm)

numCores <-detectCores()
cl <- makeCluster(numCores - 1)
registerDoParallel(cl)

# Fit survivalFM model
fit <- survivalfm::val.survivalfm(
  x = X_train_scaled,
  y = y_train,
  rank = 3,
  trace = 1,
  parallel = TRUE
)

parallel::stopCluster(cl)
```


Using a cross-validation approach for optimizing regularization parameters: 
```r
library(doParallel)
library(parallel)
library(survivalfm)

numCores <-detectCores()
cl <- makeCluster(numCores - 1)
registerDoParallel(cl)

# Fit survivalFM model
fit <- survivalfm::cv.survivalfm(
  x = X_train_scaled,
  y = y_train,
  rank = 3,
  trace = 1,
  parallel = TRUE
)

parallel::stopCluster(cl)
```

### Predicting with survivalFM

```r

# Make predictions to obtain linear predictors
lp <- survivalfm:::predict.survivalfm(fit, X_test_scaled)

# Calculate C-index on the test set
survival::concordance(y_test ~ lp, reverse = T)$concordance

````

### Accessing and visualizing model coefficients

```r

# Linear effects
linear_effects <- fit$beta %>% sort()

# Interaction effects
interaction_effects <- fit$PP

# Example visualization using pheatmap
pheatmap::pheatmap(
  mat = as.matrix(interaction_effects),
  cellheight = 10,
  cellwidth = 10
)

```
