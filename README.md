# Insurance-Claim-Expedite

Predictive Model to Expedite Selected Insurance Claims Based on Historical Company Data

Paribas Cardiff, a global insurance provider, challenged the data science community
to develop models to predict client claims that should received expedited processing.

Speedy processing is essential to client satisfaction.

The challenge was put forth on Kaggle.com:

https://www.kaggle.com/c/bnp-paribas-cardif-claims-management

Logloss was the evaluation metric for the analysis.  

I utilized boosted tree modeling using the xgboost package in R.  

The logloss achieved using optimized parameters place my results at about the top 1/3 in participants.  Combining
other models in a more complex ensemble and interpolating missing data may have improved performance.

The repository contains a file params.rda that contains optimized model parameters obtained through
a grid search with 5 fold cross validation.  The grid search was run in an AWS image with 8G RAM and 
32G memory.  In that environment, the grid search ran for approximately 72 hours.

Data pre-processing, grid search, model training, and predictions are in the file named InsuranceClaimModel.R
If you run the code I STRONGLY advise running the grid search with a small sample of training data (e.g. 100
oberservation).  If you wish to use the model to make high-quality prediction I recommend running the 
pre-processing section of the code, skipping the grid-search, loading the best parameters from the stored file,
using them to train the model using the full data set and running predictions.  Details are provided in comments
within the R file.

Project data is available at the following link:

https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/data

A blog write up I did on this model can be found at this link:

http://blog.nycdatascience.com/student-works/bnp-paribas-expediting-the-insurance-claim-process/

If you have questions contact me at:

Matt Samelson
mksamelson@gmail.com