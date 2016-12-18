
library(Amelia)
library(Hmisc)
library(mi)
library(mice)
library(VIM)
library(xgboost)


setwd("c:/users/Matt/Dropbox/DataScienceBootcamp/Projects/Capstone/")

train = read.csv("train.csv", header = TRUE,stringsAsFactors = FALSE)
test = read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE)

dim(train)
dim(test)
str(train)
str(test)
summary(train)

#Drop first two columns (index and target) from train dataframe

target <- train$target
train <-train[-c(1,2)]

#Drop first column (index) from test dataframe

test.id = test$ID
test = test[-c(1)]

#Add identifer column to each data set

train["identifier"] = 1
test["identifier"] = 2

# Missingness

#Find pct of missing data elements

pMiss <- function(x){sum(is.na(x))/length(x)*100}
pct_miss_cols = apply(train,2,pMiss)
pct_miss_rows = apply(train,1,pMiss)

#Print histograms of missing rows and columns

hist(pct_miss_rows)
hist(pct_miss_cols)  

#Generate a table showing pattern of missing values (mice)

md.pattern(train)

#Generate a nice plot and list of Missing Variables (VIM)

aggr_plot <- aggr(train, col=c('navyblue','red'), numbers=TRUE, 
                  sortVars=TRUE, labels=names(train), plot = TRUE,
                  cex.axis=.5, gap=3, cex.numbers=0.25,
                  ylab=c("Histogram of Missing Data","Pattern"))

aggr_plot


#bind train and test for certain preprocessing

preproc.df = rbind(train,test)
preproc.df$identifier = as.integer(preproc.df$identifier)

preproc.df = subset(preproc.df,,-c(v22)) #drop v22 which has 18000 levels

#Change factor variables to integers

for (f in names(preproc.df)) {
  if (class(preproc.df[[f]])=="character") {
    levels <- unique(c(preproc.df[[f]]))
    preproc.df[[f]] <- as.integer(factor(preproc.df[[f]], levels=levels))
  }
}


#Fill NAs

preproc.df[is.na(preproc.df)] = -999

#Split train and test sets again since pre-processing was complete.

train = preproc.df[preproc.df$identifier==1,]
train = train[-length(train)]#drop identifier column
train = cbind(train,target)

test = preproc.df[preproc.df$identifier==2,]
test = test[-length(test)] #drop identifier column

target = ifelse(train==0,"No","Yes")

train.sparse = sparse.model.matrix(target ~ ., data = train)
train.dmatrix <- xgb.DMatrix(data=train.sparse, label=train$target)

# set up hyper-parameter grid search with cross-validation

#NOTE:  DO NOT run the grid search with the full data training set.
#If you wish to run the code select a sample from the training set of 100 observations.
#Grid search with full training set was run in an AWS instance with 8G RAM
#and 32G memory.  Run time was approximately 72 hours.  

#Skip to line 160 to see a print of optimal parameters

#Skip to line 221 to load saved optimal parameters and trained model

nrounds.list = c(500,1000,1500,2000)
max_depth.list = c(4, 6, 8, 10)
eta.list = c(0.01, 0.001, 0.0001)
colsample_bytree.list = c(1, .8, .5)

best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (nrounds in nrounds.list){
  for (max_depth in max_depth.list){
    for (eta in eta.list){
      for (colsample_bytree in colsample_bytree.list){
        
        param <- list(objective = "binary:logistic", 
                      booster = "gbtree",
                      eval_metric = "logloss",
                      max_depth = max_depth,
                      eta = eta,
                      gamma = 1, 
                      subsample = 1,
                      colsample_bytree = colsample_bytree, 
                      min_child_weight = 1,
                      max_delta_step = 1)
        
        cv.nfold = 5
        seed.number = sample.int(10000, 1)[[1]]
        set.seed(seed.number)
        
        mdcv <- xgb.cv(data=train.dmatrix, params = param, nthread=6, 
                       nfold=cv.nfold, nrounds = nrounds,
                       verbose = T,  maximize=FALSE)
        
        min_logloss = min(mdcv[, test.logloss.mean])
        min_logloss_index = which.min(mdcv[, test.logloss.mean])
        
        if (min_logloss < best_logloss) {
          best_logloss = min_logloss
          best_logloss_index = min_logloss_index
          best_seednumber = seed.number
          best_param = param
          }
        
      }
    }
  }
}

#Save optimal parameters, minimum logloss, for later use. 
#NOTE - Save command for parameters are commented out to avoid
#accidental overwrite.

#save(nround,best_seednumber,best_param,file="params.Rda")
load("params.Rda")

#Print of best parameters, optimal number of model rounds, and minimum overall log-loss

nround = best_logloss_index
nround
best_seednumber
best_param
best_logloss

# > nround
# [1] 1312
# > best_seednumber
# [1] 3515
# > best_param
# $objective
# [1] "binary:logistic"
# 
# $booster
# [1] "gbtree"
# 
# $eval_metric
# [1] "logloss"
# 
# $max_depth
# [1] 10
# 
# $eta
# [1] 0.01
# 
# $gamma
# [1] 1
# 
# $subsample
# [1] 1
# 
# $colsample_bytree
# [1] 0.5
# 
# $min_child_weight
# [1] 1
# 
# $max_delta_step
# [1] 1
# 
# > best_logloss
# [1] 0.460109
# >

#Train the model using the optimal parameters and full training set

set.seed(best_seednumber)
md <- xgb.train(data=train.dmatrix, params=best_param, nrounds=nround, nthread=6)

#Load the best parameters and the fully trained model

#md <- xgb.load('xgb.model')

#Prepare test set for prediction generation and predict targets for submission to Kaggle

test.f = sparse.model.matrix(~ ., data = test)
set.seed(0)
preds.test = predict(md, newdata= test.f)

submission <- data.frame(ID=test.id, PredictedProb=preds.test)

write.csv(submission, "submissions.csv",row.names = FALSE)

#Logloss on test set was .45664 as reported by Kaggle.  This would
#have placed approximately in the top 1/3 of competitors in the competition

#Assess feature importance

names <- dimnames(train)[[2]][-length(train)]
importance_matrix <- xgb.importance(names,model = md)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix[1:10,])

#Feature v51 is by far the most important model feature (see image)

#Print Model

xgb.plot.tree(model = md)
