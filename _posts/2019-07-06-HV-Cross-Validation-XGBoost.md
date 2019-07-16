---
layout: single
title:  "Cross Validation in Time Series: An example of HV-Block CV with XGBoost"
date:   2019-07-06 16:31:24 +0200
categories: R
# classes: normal
author_profile: true
---

In order to estimate the predictive performance of a supervised machine learning model, we usually split the available data in multiple chunks for training, validating and testing the model.
There are several techniques that can be used to estimate the performance of a model, being cross validation one of the most common.

We can carry out cross validation in some different ways. Here I am showing some of them, and discussing the implications they may have when using with temporal data

## Splitting the data

In all figures below I am using this notation to split training and validation sets:

![Train & test notation](/assets/images/train_test_notation.png){:class="image-centered"}

Maybe the most common way to split the data is this way (for a 5-fold cross validation):

![Normal CV](/assets/images/train_test_normal.png){:class="image-centered"}

But, what if the data we are dealing with consists of one or several time series?

The natural approach consists in splitting the data chronologically, using just the past as training observations and forecasting into the future. This technique is usually called walk-forward validation and here is an sketch:

![TS CV](/assets/images/train_test_ts.png){:class="image-centered"}

That could be an option to cross validate your model.

## Validating the model

When developing a backtesting strategy, the main thing you should be focusing on is simulating real predicting conditions as close as possible.
You don't want to fool yourself, so you should be very careful when preparing and splitting your data.

Misusing the validation technique or designing an inappropriate backtesting strategy could lead to wrongly estimate the performance of the model.

This could be a common situation when working with highly autocorrelated data like a time series.

Why? Because in a time series, points are usually similar to their neighbours, and therefore using one observation $$ (Y_t, X_t) $$ to train the model and the contiguous one $$ (Y_{t+1}, X_{t+1}) $$ to evaluate it usually leads to **data leakage**.
<!-- --- Knowing $$ Y_t $$ makes $$ Y_{t+1} $$ *easier* to predict. -->
<!-- or ---If you know the relationship between $$ X_{t,i} $$ and $$ Y_t $$, it's easier to know $$ Y_{t+1} $$ given $$ X_{t+1,i} $$, so contiguous observations belonging to training and validation set respectively easily creates leaking.  -->


## Data leakage

Data leakage happens when information from outside the training set is used to build the model.
This can occur in several ways.

It may happen explicitly, for example if you happen to have duplicated observations and use some of them to train and some to test the model. Generally this would be an scenario you would try to avoid, unless you have enough domain knowledge to foresee that such an observation will happen again in the future.

Sometimes it happens in a more implicit way, for example when normalizing the data, imputing missing values or creating label-encoded features before splitting the dataset to cross validate.
In those scenarios the shit is real: You have leaked information from outside the training set to build the model.

There is even a subtler case of data leakage: Using all the data to select the features that will be used to build the model and then evaluating it with part of it will cause leakage too.

> When developing a backtesting strategy, the main thing you should be focusing on is simulating real predicting conditions as close as possible.

More generally, if you are using some kind of cross validation technique you should do all the preparation stuff, *within each fold*.
In our case, as we said before, we should be very careful when splitting our data set because of the autocorrelation of both our features and target.


<!-- ---Moreover, if we were using some lags of the regressors as features, we would be *leaking* information into the test folds, as some data that is being used to train the model is also part of the test set.--- -->

## Avoiding data leakage

So, coming back to the original question:
How could we reliably validate our model?

> When the amount of available data to train the model is scarce you must be creative.

The chronological approach showed above is valid as long as we don't leakage any test data and we replicate the situation we will have when forecasting.

What information will I have when I have to make the predictions? Will I know $$ Y_t $$ when forecasting $$ Y_{t+1} $$?

Sometimes you have to forecast several hours or days ahead, and so you are interested in the predictive capacity of the model in that range of horizons.
For such situations, we could introduce a *gap* in our validation strategy, in grey in our picture below:

![TS CV](/assets/images/train_test_ts_gap.png){:class="image-centered"}

Maybe the main drawback of this strategy is that it doesn't make full use of the available data.
When the amount of available data to train the model is scarce you must be creative in order to get a proper estimation of the model performance.

Truth is, it isn't written anywhere that you can't use future information to make predictions into the past. In most cases you are allowed to do it, but this may depend on the problem.

So actually, we *could* use the standard cross validation approach instead, slightly modified with the *gap* concept

## HV-Block Cross Validation

One approach that tries to make full use of the available data while preserving our model from using *forbidden* information is called **HV-Block Cross Validation**.

This technique consists of dropping from the training data some observations that are too close to the test data. That is, manually creating a gap in your data between both data sets:

![HV CV](/assets/images/train_test_hvcv.png){:class="image-centered"}

Ideally, the size of the *gap* should be the first Y lag that doesn't show any autocorrelation.

I have developed some R code that allow us to perform this type of cross validation with the famous *XGBoost* library.
Full code is presented below.

The idea was to create a function that offered us the same functionality as the built-in *xgb.cv* function, but that performs the validation holding out some data between the train and the
validation folds (the gap).
Both the training and the evaluation of the predictions are done manually inside each step of the loop, corresponding to each fold.


## XGBoost HVCV function

In this first chunk we are just loading the libraries and creating the dataset:

```r
# Loading required packages

library(tidyverse)
library(lubridate)
library(magrittr)
library(xgboost)

# Creating dataset  (must be ordered)
trainSet <- tibble(DateTime = seq(ymd_h(2019050100),
                                  ymd_h(2019050100) %>% add(hours(50-1)),
                                  by = "hours"),
                   y = rnorm(50),
                   x1 = rnorm(50),
                   x2 = rnorm(50)) %>%
  arrange(DateTime)

nFolds <- 4
}
```

Then, we divide the data chronologically into folds and store the row indexes belonging to each fold in a list:

```r
customFolds <- c(rep(1:nFolds, each = nrow(trainSet)/nFolds),
                 sample(1:nFolds, size = nrow(trainSet) - round(nrow(trainSet)/nFolds)*nFolds)) %>%
  sort() %>%
  enframe(name = "Row", value = "Fold") %>%
  group_by(Fold) %>%
  group_split(keep = FALSE) %>%
  map(~ .x %>% flatten_int()) %>%
  set_names(paste0("Fold", 1:nFolds))
}
```

*customFolds* contains ordered row indexes assigned to different folds. This is how it looks:

```r
print(customFolds)

$Fold1
 [1]  1  2  3  4  5  6  7  8  9 10 11 12

$Fold2
 [1] 13 14 15 16 17 18 19 20 21 22 23 24 25

$Fold3
 [1] 26 27 28 29 30 31 32 33 34 35 36 37 38

$Fold4
 [1] 39 40 41 42 43 44 45 46 47 48 49 50
```

And then, the main function.
**XGB_HVCV** is a wrapper of the xgboost function *xgb.train*, and its console output is similar to the *xgb.cv* function.

```r
XGB_HVCV <- function(originalTrainSet, customFolds, hoursGap, verbose = F, eval_metric = "rmse",      
                     print_every_n = 50, early_stopping_rounds = 5, maxRounds = 5000,
                     params = list(eta = .1, max_depth = 4, gamma = 2, alpha = 0.5, lambda = 1,
                                   colsample_bytree = 0.7, min_child_weight = 6, subsample = .8)){

  # Functions expects:

  # originalTrainSet is a tibble with
  ## Time-index called 'DateTime' in column 1
  ## Label in column 2

  # customFolds is a list of length the number of folds, with the row indexes as a vector in each element
  # hoursGap is a positive integer

  trainingError <- validationError <- nRounds <-  rep(NA, length(customFolds))

  for(f in 1:length(customFolds)){

    validationSet <- originalTrainSet[customFolds[[f]],]

    trainSet <- anti_join(originalTrainSet, validationSet, by = "DateTime") %>%
      arrange(DateTime) %>%
      filter(!(between(DateTime,
                       last(validationSet$DateTime) %>% add(hours(1)),
                       last(validationSet$DateTime) %>% add(hours(1)) %>% add(hours(hoursGap))))) %>%
      filter(!(between(DateTime,
                       first(validationSet$DateTime) %>% add(hours(-1)) %>% add(hours(-hoursGap)),
                       first(validationSet$DateTime) %>% add(hours(-1)))))


    trainSet_xgb <- xgb.DMatrix(data.matrix(trainSet[,-c(1,2)]), label = data.matrix(trainSet[,2]))
    validationSet_xgb <- xgb.DMatrix(data.matrix(validationSet[,-c(1,2)]), label = data.matrix(validationSet[,2]))

    watchlist <- list(train = trainSet_xgb,
                      validation = validationSet_xgb)

    xgbModel <- xgb.train(data = trainSet_xgb,
                          params = params,
                          verbose = verbose,
                          maximize = FALSE,
                          print_every_n = print_every_n,
                          nrounds = maxRounds,
                          watchlist = watchlist,
                          early_stopping_rounds = early_stopping_rounds,
                          callbacks = list(cb.evaluation.log()))


    nRounds[f] <- xgbModel$best_iteration
    trainingError[f] <- xgbModel$evaluation_log[[paste0("train_", eval_metric)]][xgbModel$best_iteration]
    validationError[f] <- xgbModel$best_score
  }


  meanTrainError <- round(mean(trainingError),3)
  meanValidationError <- round(mean(validationError),3)
  bestIteration <- round(mean(nRounds),3)

  if(verbose){
    cat("*****************************************************\n")
    cat("\n")
    cat("Global results: \n")
    cat(sprintf("train-%s:%s test-%s:%s \n",
                eval_metric, meanTrainError, eval_metric, meanValidationError))
    cat(sprintf("bestIter = %s \n", bestIteration))
    cat("\n")
    cat("\n")
  }

  crossValidationResults <- tibble(ValidationFold = 1:length(customFolds),
                                   TrainingError = trainingError,
                                   ValidationError = validationError,
                                   BestIteration = nRounds) %>%
    add_row(ValidationFold = "Global",
            TrainingError = meanTrainError,
            ValidationError = meanValidationError,
            BestIteration = bestIteration)

  return(crossValidationResults)
}
```

The function returns a tibble summarizing the in-sample and out-of-sample errors in each fold:

```r
# A tibble: 5 x 4
  ValidationFold TrainingError ValidationError BestIteration
  <chr>                  <dbl>           <dbl>         <dbl>
1 1                      0.767           0.765            52
2 2                      0.625           1.19             80
3 3                      0.694           0.699            61
4 4                      0.981           1.50             26
5 Global                 0.767           1.04             55
```


## References
  - [HV-Block Cross-Validation paper](https://pdfs.semanticscholar.org/2b03/7be3e9a97c2e720da547e9bea57edd31aec6.pdf)
  - [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/)
