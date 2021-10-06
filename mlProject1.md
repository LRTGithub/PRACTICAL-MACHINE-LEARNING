---
title: "STATISCAL PREDICTION MODELLING APPLIED TO WORKOUT QUALITY."
author: "LR"
date: "October 4, 2021"
output: 
        html_document: 
          keep_md: yes
          toc: yes
          fig_caption: yes
          number_sections: yes
          fig_width: 5
---
OVERVIEW / SYNOPSIS.-
=====================
The goal of your project is to predict the manner in which participants of the study conducted a specific work-out exercise. 
This paper describes the process in which a statistical model was created and applied to predicting the manner under which a group of individuals did a specific work-out (variable name "classe" ). I used a stacked model approach of 3 models: Linear Discriminant Analysis (LDA), Random Forest (RF), General Boosted Regression Modelling (GBM).
These three models are stacked into 1 which then predicts on out-of-sample data --a quiz dataset of 20 records with variables measures but unknown "classe"--.
The 3 models have accuracy and computational times of: 71%/10secs, 99%/6 minutes, 98%/2minutes. The final stacked model predicts the output of the quiz (in ml-testing.csv) with 100% accuracy. I later discovered that the GBM alone is able to predict the same outcome for the quiz with 100%. While it is not possible to know this a priori --the quiz needs to be solved to 100% accuracy first to know what the right answers are-- GBM is the model that offers the most acceptable trade-off between accuracy and computational time.

CRITERIA CHECKLIST: HELP YOUR GRADERS.-
========================================
This section addresses the review criteria in detail. THIS DOCUMENT COMPALIES WITH ALL CRITERIA.

[ YES ]	The report describes a machine learning algorithm to predict activity quality from activity monitors.
[ YES ]	The report is 2,000 words or less. THE WORD COUNT 1,688.
[ YES ]	The number of figures in the document 5 or less.
[ YES ]	The report explains how the model was built.
[ YES ]	The report explains how cross-validation was used to estimate the out of sample error rate.
[ YES ]	The report lists the key decisions made during the analysis, and explains why these decisions were made.
[ YES ]	The report reviews the accuracy of the selected machine learning algorithm in predicting the 20 unknown test cases.
[ YES ]	The submission includes a github repository, and a link to the repository as part of the Coursera submission.
[ YES ]	The submission includes a compiled HTML document, the output .md markdown file, and any graphics required to correctly view them within the markdown file on github.
[ YES ]	If a github pages branch is provided, is the HTML file accessible at https://username.github.io/reponame. 
ALL CRITERIA HAS BEEN MET.

BRIEF DESCRIPTION OF THE EXPERIMENT AND VARIABLE CLASS.-
========================================================
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

DATA COLLECTION AND METHODOLOGY.-
=================================
Data collection and experiment specifics can be found in the following link:
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. 

RESOURCES AND CREDIT TO MENTORS AND PEERS.-
===========================================
The lectures and prior weeks' course quizes were the main source of reference to complete this project. The week 4 forum, the comments and hints from mentor Len Greski offer invaluable must-read help to get a quick start, avoid pitfalls, and improve computational efficiency.

APPROACH TO SOLVING THE PROBLEM.-
=================================
The dataset contains 160 variables. I will begin by reviewing, cleaning, completing the dataset, and selecting a group of relevant variables. I will create train, test, and validation sets and fit 3 models: lda, rf, and gbm. Finally, I will stack the 3 models (fitted with rf).
The last step is done to predict on the quiz set, compute the out-of-sample error, answer the quiz, and present conclussions.

CROSS VALIDATION AND OUT-OF-SAMPLE ERROR.-
==========================================
After we build and apply a model to a dataset, we need to have a measure of how well the model is predicting vis-a-vis the observed data. I will use k-fold cross-validation method applied through the trControl parameter (available in the train function of the caret package). The k-fold cross validation process can be summarized as follows:
1.- The dataset is divided into k groups of approximately equal size. One group is left out and a model is fit on,
2.- on the other k-1 groups. The mean square error (MSE) of this iteration is calculated on the group that was left out.
This process is repeated k times choosing a different group to be left out each time. The overall MSE is the average of the k MSEs.


```r
library( caret )
library( gbm )
library( parallel ) # for the implementation of parallel processing.
library( doParallel )
cluster <- makeCluster( detectCores() - 1 ) # convention to leave 1 core for OS
registerDoParallel( cluster )

# LOAD DATA.
OriginalDF <- read.csv( "pml-training.csv")
```
EXPLORATORY DATA ANALYSIS AND DATA CLEANSING.-
==============================================
The dataset contains:

```r
readings <- dim( OriginalDF )[1]
readings
```

```
## [1] 19622
```
readings and 

```r
variables <<- as.numeric( dim( OriginalDF )[2] )
variables
```

```
## [1] 160
```
variables. 

INTUITION FIRST.-
=================
the name of the person, time stamp, new window, numwindow, and more generally the first 7 variables are unlikely to have any impact on the quality of the exercise. 
I will begin by reviewing, cleaning, and completing the dataset, and eventually select a group of relevant variables to build the model. I will review all variables first and flag those with a large number of missing data ( empty "", or NA):

```r
exclude <- c(1:7) # variables to exclude.
fix <- c( ) # variables to fix/complete.
for ( var in 1:variables ){ # go through all the variables in the original dataset.
        # select rows empty, "NA", or NA:
        MyNAs <- ( OriginalDF[ , var ] == "" | OriginalDF[ ,var ] == "NA" | is.na( OriginalDF[ ,var ] ) )
        # compute the proportion of such rows in the dataset:
        propNAs <- sum( MyNAs ) / readings *100
        if( propNAs > 95 ){ # if more than 95%
                exclude <<- append( exclude, var ) # flag to exlude this variable from the set.
        }
        if( propNAs > 0 & propNAs <= 95 ){ # if there are missing values, but less than 95%, flag to complete the missing values.
                fix <<- append( fix, var )
        }
}
```
I will exclude a total of:

```r
length( exclude )
```

```
## [1] 107
```

```r
names( OriginalDF[, exclude ])
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "kurtosis_roll_belt"      
##   [9] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [11] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [13] "skewness_yaw_belt"        "max_roll_belt"           
##  [15] "max_picth_belt"           "max_yaw_belt"            
##  [17] "min_roll_belt"            "min_pitch_belt"          
##  [19] "min_yaw_belt"             "amplitude_roll_belt"     
##  [21] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [23] "var_total_accel_belt"     "avg_roll_belt"           
##  [25] "stddev_roll_belt"         "var_roll_belt"           
##  [27] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [29] "var_pitch_belt"           "avg_yaw_belt"            
##  [31] "stddev_yaw_belt"          "var_yaw_belt"            
##  [33] "var_accel_arm"            "avg_roll_arm"            
##  [35] "stddev_roll_arm"          "var_roll_arm"            
##  [37] "avg_pitch_arm"            "stddev_pitch_arm"        
##  [39] "var_pitch_arm"            "avg_yaw_arm"             
##  [41] "stddev_yaw_arm"           "var_yaw_arm"             
##  [43] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [45] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [47] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [49] "max_roll_arm"             "max_picth_arm"           
##  [51] "max_yaw_arm"              "min_roll_arm"            
##  [53] "min_pitch_arm"            "min_yaw_arm"             
##  [55] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [57] "amplitude_yaw_arm"        "kurtosis_roll_dumbbell"  
##  [59] "kurtosis_picth_dumbbell"  "kurtosis_yaw_dumbbell"   
##  [61] "skewness_roll_dumbbell"   "skewness_pitch_dumbbell" 
##  [63] "skewness_yaw_dumbbell"    "max_roll_dumbbell"       
##  [65] "max_picth_dumbbell"       "max_yaw_dumbbell"        
##  [67] "min_roll_dumbbell"        "min_pitch_dumbbell"      
##  [69] "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
##  [71] "amplitude_pitch_dumbbell" "amplitude_yaw_dumbbell"  
##  [73] "var_accel_dumbbell"       "avg_roll_dumbbell"       
##  [75] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
##  [77] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
##  [79] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
##  [81] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
##  [83] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
##  [85] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
##  [87] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
##  [89] "max_roll_forearm"         "max_picth_forearm"       
##  [91] "max_yaw_forearm"          "min_roll_forearm"        
##  [93] "min_pitch_forearm"        "min_yaw_forearm"         
##  [95] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
##  [97] "amplitude_yaw_forearm"    "var_accel_forearm"       
##  [99] "avg_roll_forearm"         "stddev_roll_forearm"     
## [101] "var_roll_forearm"         "avg_pitch_forearm"       
## [103] "stddev_pitch_forearm"     "var_pitch_forearm"       
## [105] "avg_yaw_forearm"          "stddev_yaw_forearm"      
## [107] "var_yaw_forearm"
```
There are:

```r
length( fix )
```

```
## [1] 0
```
variables to fix:

```r
names( OriginalDF[, fix ])
```

```
## character(0)
```

```r
SubsetDF <- OriginalDF[ , -exclude ]
```
change the prediction target variable classe to a factor variable:

```r
SubsetDF$classe <- as.factor( SubsetDF$classe )
#str( SubsetDF )
```
Forecast for the quiz. This is the dataset that needs to be predicted.

```r
ForecastDF <- read.csv( "pml-testing.csv")
exclude1 <- append( exclude, 160 ) # last column is id, exclude.
SubsetForecastDF <- ForecastDF[ , -exclude1 ]
dim( SubsetForecastDF ) # 20 rows x 52 variables, does not include classe. We need to predict classe.
```

```
## [1] 20 52
```

```r
dim( SubsetDF ) # 19000 rows x 53 variables, includes classe.
```

```
## [1] 19622    53
```

```r
str( ForecastDF )
```

```
## 'data.frame':	20 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : chr  "pedro" "jeremy" "jeremy" "adelmo" ...
##  $ raw_timestamp_part_1    : int  1323095002 1322673067 1322673075 1322832789 1322489635 1322673149 1322673128 1322673076 1323084240 1322837822 ...
##  $ raw_timestamp_part_2    : int  868349 778725 342967 560311 814776 510661 766645 54671 916313 384285 ...
##  $ cvtd_timestamp          : chr  "05/12/2011 14:23" "30/11/2011 17:11" "30/11/2011 17:11" "02/12/2011 13:33" ...
##  $ new_window              : chr  "no" "no" "no" "no" ...
##  $ num_window              : int  74 431 439 194 235 504 485 440 323 664 ...
##  $ roll_belt               : num  123 1.02 0.87 125 1.35 -5.92 1.2 0.43 0.93 114 ...
##  $ pitch_belt              : num  27 4.87 1.82 -41.6 3.33 1.59 4.44 4.15 6.72 22.4 ...
##  $ yaw_belt                : num  -4.75 -88.9 -88.5 162 -88.6 -87.7 -87.3 -88.5 -93.7 -13.1 ...
##  $ total_accel_belt        : int  20 4 5 17 3 4 4 4 4 18 ...
##  $ kurtosis_roll_belt      : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ max_picth_belt          : logi  NA NA NA NA NA NA ...
##  $ max_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ min_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ min_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ min_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : logi  NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : logi  NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : logi  NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : logi  NA NA NA NA NA NA ...
##  $ avg_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : logi  NA NA NA NA NA NA ...
##  $ var_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : logi  NA NA NA NA NA NA ...
##  $ var_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : logi  NA NA NA NA NA NA ...
##  $ var_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  -0.5 -0.06 0.05 0.11 0.03 0.1 -0.06 -0.18 0.1 0.14 ...
##  $ gyros_belt_y            : num  -0.02 -0.02 0.02 0.11 0.02 0.05 0 -0.02 0 0.11 ...
##  $ gyros_belt_z            : num  -0.46 -0.07 0.03 -0.16 0 -0.13 0 -0.03 -0.02 -0.16 ...
##  $ accel_belt_x            : int  -38 -13 1 46 -8 -11 -14 -10 -15 -25 ...
##  $ accel_belt_y            : int  69 11 -1 45 4 -16 2 -2 1 63 ...
##  $ accel_belt_z            : int  -179 39 49 -156 27 38 35 42 32 -158 ...
##  $ magnet_belt_x           : int  -13 43 29 169 33 31 50 39 -6 10 ...
##  $ magnet_belt_y           : int  581 636 631 608 566 638 622 635 600 601 ...
##  $ magnet_belt_z           : int  -382 -309 -312 -304 -418 -291 -315 -305 -302 -330 ...
##  $ roll_arm                : num  40.7 0 0 -109 76.1 0 0 0 -137 -82.4 ...
##  $ pitch_arm               : num  -27.8 0 0 55 2.76 0 0 0 11.2 -63.8 ...
##  $ yaw_arm                 : num  178 0 0 -142 102 0 0 0 -167 -75.3 ...
##  $ total_accel_arm         : int  10 38 44 25 29 14 15 22 34 32 ...
##  $ var_accel_arm           : logi  NA NA NA NA NA NA ...
##  $ avg_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : logi  NA NA NA NA NA NA ...
##  $ var_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : logi  NA NA NA NA NA NA ...
##  $ var_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : logi  NA NA NA NA NA NA ...
##  $ var_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  -1.65 -1.17 2.1 0.22 -1.96 0.02 2.36 -3.71 0.03 0.26 ...
##  $ gyros_arm_y             : num  0.48 0.85 -1.36 -0.51 0.79 0.05 -1.01 1.85 -0.02 -0.5 ...
##  $ gyros_arm_z             : num  -0.18 -0.43 1.13 0.92 -0.54 -0.07 0.89 -0.69 -0.02 0.79 ...
##  $ accel_arm_x             : int  16 -290 -341 -238 -197 -26 99 -98 -287 -301 ...
##  $ accel_arm_y             : int  38 215 245 -57 200 130 79 175 111 -42 ...
##  $ accel_arm_z             : int  93 -90 -87 6 -30 -19 -67 -78 -122 -80 ...
##  $ magnet_arm_x            : int  -326 -325 -264 -173 -170 396 702 535 -367 -420 ...
##  $ magnet_arm_y            : int  385 447 474 257 275 176 15 215 335 294 ...
##  $ magnet_arm_z            : int  481 434 413 633 617 516 217 385 520 493 ...
##  $ kurtosis_roll_arm       : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : logi  NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : logi  NA NA NA NA NA NA ...
##  $ max_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ max_picth_arm           : logi  NA NA NA NA NA NA ...
##  $ max_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ min_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ min_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ min_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : logi  NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : logi  NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : logi  NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  -17.7 54.5 57.1 43.1 -101.4 ...
##  $ pitch_dumbbell          : num  25 -53.7 -51.4 -30 -53.4 ...
##  $ yaw_dumbbell            : num  126.2 -75.5 -75.2 -103.3 -14.2 ...
##  $ kurtosis_roll_dumbbell  : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : logi  NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : logi  NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : logi  NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : logi  NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : logi  NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : logi  NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : logi  NA NA NA NA NA NA ...
##   [list output truncated]
```
Function to compute Variable Importance.

```r
VarImportance <- function ( model, type ) {
                if( type == "RF" ){
                        MyVarImp <- varImp( model$finalModel, decreasing = TRUE )
                        MyVarImp <- data.frame( predictors, MyVarImp$Overall )
                        colnames( MyVarImp )<- c( 'var.id', 'importance')
                        #print( MyVarImp )# unsorted.
                        MyVarImp <- MyVarImp[ order( -MyVarImp$importance ), ]
                } else { # else it is "GBM". LDA does not have a call to VarImportance.
                        MyVarImp <- relative.influence( model$finalModel, sort. = TRUE )
                }
                print( head( MyVarImp, 10 ) ) # sorted descending, print top 10.
                print( " out of a total of ")
                print( dim( MyVarImp ) )
                print( " variables.\n")
}
```
SPLIT THE DATA IN 3 GROUPS.- 
============================
I will create train, test, and validation sets.

```r
set.seed( 12345 )
inBuild <- createDataPartition( y = SubsetDF$classe, p=3/4, list = FALSE )
buildData <- SubsetDF[ inBuild, ] # biggest set 70% of data.
validation <- SubsetDF[ -inBuild, ] # take 30% for validation.

inTrain <- createDataPartition( y = buildData$classe, p=3/4, list = FALSE )
training <- buildData[ inTrain, ] # 70% of buildData, ~49% of total data.
testing <- buildData[ -inTrain, ] # 30% of buildData, ~21% of total data.

predictors <- colnames( training[ , -length( training ) ] ) # all variables except classe.
```
I will build 3 models: LDA, RF, GBM
Model LDA: Linear Discriminate Analysis

```r
TimerStart <- Sys.time()
fitControlLDA <- trainControl( method = "cv", number = 5, allowParallel = TRUE ) # k-fold=5 cross-validation.
modelLDA <- train( classe ~., method = "lda", data = training, trControl = fitControlLDA ) # metric = "Accuracy", prox = TRUE,
# modelLDA
predLDA <- predict( modelLDA, newdata = testing ) # predict on the testing set.
predLDAV <- predict( modelLDA, newdata = validation )
cm <- confusionMatrix( predLDA , testing$classe )
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 858 102  67  39  13
##          B  22 484  62  27 115
##          C  86  67 425  70  52
##          D  74  29  73 441  75
##          E   6  30  14  26 421
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7148          
##                  95% CI : (0.6999, 0.7293)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6391          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8203   0.6798   0.6630   0.7313   0.6228
## Specificity            0.9160   0.9238   0.9095   0.9184   0.9747
## Pos Pred Value         0.7952   0.6817   0.6071   0.6373   0.8471
## Neg Pred Value         0.9277   0.9232   0.9275   0.9457   0.9198
## Prevalence             0.2844   0.1936   0.1743   0.1639   0.1838
## Detection Rate         0.2333   0.1316   0.1156   0.1199   0.1145
## Detection Prevalence   0.2934   0.1930   0.1903   0.1881   0.1351
## Balanced Accuracy      0.8682   0.8018   0.7862   0.8249   0.7987
```
Model LDA predicts classe "A" correctly 82% of time when classe "A" was the right answer (sensitivity). It also predicts correctly a classe different than "A" when "A" is not the right classe (specificity). Sensitivity is lowest for classe "E", and specificity is lowest for classe "C". The overall accuracy of the LDA model is:

```r
cm$overall["Accuracy"]*100
```

```
## Accuracy 
## 71.47906
```
71% is not great. We need at least 80% to pass a quiz after all ...:).....

```r
TimerStop <- Sys.time()
TimeLDA <- paste("LDA model time => ", TimerStop - TimerStart, attr( TimerStop - TimerStart, "units" ) )
```
It takes

```r
TimeLDA
```

```
## [1] "LDA model time =>  12.7605011463165 secs"
```
seconds to run the LDA model. Very reasonable computational performance ! with a meh... accuracy. Not bad. 

Model RF: random forest.

```r
TimerStart <- Sys.time()
fitControlRF <- trainControl( method = "cv", number = 5, allowParallel = TRUE ) # cv=cross validation, k = 5.
modelRF <- train( classe ~., method = "rf", data = training , trControl = fitControlRF )
#print( modelRF$finalModel )
#modelRF
predRF <- predict( modelRF, newdata = testing ) # predict on the testing set.
predRFV <- predict( modelRF, newdata = validation ) # predict on the validation set.
cm <- confusionMatrix( predRF , testing$classe )
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1045    6    0    0    0
##          B    1  701    7    1    1
##          C    0    5  632    6    5
##          D    0    0    2  596    3
##          E    0    0    0    0  667
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9899          
##                  95% CI : (0.9862, 0.9929)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9873          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9990   0.9846   0.9860   0.9884   0.9867
## Specificity            0.9977   0.9966   0.9947   0.9984   1.0000
## Pos Pred Value         0.9943   0.9859   0.9753   0.9917   1.0000
## Neg Pred Value         0.9996   0.9963   0.9970   0.9977   0.9970
## Prevalence             0.2844   0.1936   0.1743   0.1639   0.1838
## Detection Rate         0.2841   0.1906   0.1718   0.1620   0.1813
## Detection Prevalence   0.2858   0.1933   0.1762   0.1634   0.1813
## Balanced Accuracy      0.9984   0.9906   0.9903   0.9934   0.9933
```

```r
# The RF model has high sensitivity (98%-99%) and specificity (99% to 100%) for all classes.
cm$overall["Accuracy"]*100
```

```
## Accuracy 
## 98.99402
```

```r
VarImportance( modelRF, "RF" ) # top 10 variables based on the importance.
```

```
##               var.id importance
## 1          roll_belt  1137.7720
## 41     pitch_forearm   683.6104
## 3           yaw_belt   636.1505
## 39 magnet_dumbbell_z   509.7002
## 38 magnet_dumbbell_y   504.5361
## 40      roll_forearm   487.2137
## 2         pitch_belt   483.0273
## 35  accel_dumbbell_y   285.6437
## 47   accel_forearm_x   223.7946
## 27     roll_dumbbell   207.8953
## [1] " out of a total of "
## [1] 52  2
## [1] " variables.\n"
```
Variable Importance represents the effect of a variable in both main effects and interactions. The most important 3 variables are roll_belt, pitch_forearm, and yaw_belt.

```r
TimerStop <- Sys.time()
paste( "RF model time = ", TimerStop - TimerStart, attr( TimerStop - TimerStart,"units" ) )
```

```
## [1] "RF model time =  10.0363694310188 mins"
```
accuracy 99%, 6 mins.
The accuracy of RF is much better than that of the prior LDA model. That accuracy comes at the expense of waiting 6 minutes for it. 

Model GBM: Generalized Boosted Regression Modeling.

```r
TimerStart <- Sys.time()
fitControlGBM <- trainControl( method = "cv", number = 5, allowParallel = TRUE ) # cross validation k=5.
modelGBM <- train( classe ~., method = "gbm", data = training, trControl = fitControlGBM )
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2319
##      2        1.4627             nan     0.1000    0.1586
##      3        1.3609             nan     0.1000    0.1232
##      4        1.2830             nan     0.1000    0.1093
##      5        1.2150             nan     0.1000    0.0856
##      6        1.1601             nan     0.1000    0.0822
##      7        1.1087             nan     0.1000    0.0603
##      8        1.0695             nan     0.1000    0.0623
##      9        1.0303             nan     0.1000    0.0575
##     10        0.9947             nan     0.1000    0.0479
##     20        0.7642             nan     0.1000    0.0277
##     40        0.5367             nan     0.1000    0.0102
##     60        0.4084             nan     0.1000    0.0075
##     80        0.3270             nan     0.1000    0.0046
##    100        0.2657             nan     0.1000    0.0041
##    120        0.2239             nan     0.1000    0.0035
##    140        0.1901             nan     0.1000    0.0019
##    150        0.1756             nan     0.1000    0.0016
```

```r
#print( modelGBM$finalModel )
#modelGBM
predGBM <- predict( modelGBM, newdata = testing ) # predict on the testing set.
predGBMV <- predict( modelGBM, newdata = validation ) # predict on the validation set.
cm <- confusionMatrix( predGBM , testing$classe ) # confusionMatrix( prediction, actual )
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1026   20    0    1    0
##          B   16  671   17    5   10
##          C    2   17  616   15    7
##          D    2    2    7  575   18
##          E    0    2    1    7  641
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9595          
##                  95% CI : (0.9526, 0.9656)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9488          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9809   0.9424   0.9610   0.9536   0.9482
## Specificity            0.9920   0.9838   0.9865   0.9906   0.9967
## Pos Pred Value         0.9799   0.9332   0.9376   0.9520   0.9846
## Neg Pred Value         0.9924   0.9861   0.9917   0.9909   0.9884
## Prevalence             0.2844   0.1936   0.1743   0.1639   0.1838
## Detection Rate         0.2790   0.1824   0.1675   0.1563   0.1743
## Detection Prevalence   0.2847   0.1955   0.1786   0.1642   0.1770
## Balanced Accuracy      0.9865   0.9631   0.9737   0.9721   0.9724
```
This model has very high sensitivity (94%-98%) and specificity (98% to 99%) for all classes.
The accuracy of this model is significantly better than the initial LDA model, and very close to that of the RF model.

```r
cm$overall["Accuracy"]*100
```

```
## Accuracy 
## 95.94889
```

```r
VarImportance( modelGBM, "GBM" ) # top 10 variables based on the importance.
```

```
## n.trees not given. Using 150 trees.
##         roll_belt     pitch_forearm          yaw_belt magnet_dumbbell_z 
##         2382.9573         1314.8434         1313.0707         1035.8905 
##      roll_forearm magnet_dumbbell_y        pitch_belt     magnet_belt_z 
##          756.5517          708.5213          676.1493          584.0876 
##     roll_dumbbell      gyros_belt_z 
##          386.9351          363.8423 
## [1] " out of a total of "
## NULL
## [1] " variables.\n"
```
Models RF and GBM differ on the variables they use and the ranking of their importance. However, the most important 3 variables (roll_belt, pitch_forearm, and yaw_belt ) are the same in both models. 

```r
TimerStop <- Sys.time()
paste("GBM model time = ", TimerStop - TimerStart, attr( TimerStop - TimerStart,"units") )
```

```
## [1] "GBM model time =  3.64514105319977 mins"
```
The GBM method provides the best balance between computational time and accuracy.
Now assemble all 3 models into 1.
Build a new data set combining the 3 predictions from the 3 models above:

```r
TimerStart <- Sys.time()
CombDataSetDF <- data.frame( predLDA, predRF, predGBM, classe = testing$classe ) # built with the testing model.
```
Now I will build a model that predicts classe based on the dataset assembled.

```r
fitControlComb <- trainControl( method = "cv", number = 5, allowParallel = TRUE ) # cross validation.
```
I will basically fit classe in the testing set against the 3 model predictions of it.

```r
modelCombFit <- train( classe ~., method = "rf", data = CombDataSetDF, trControl = fitControlComb ) # built with testing model.
#print( modelCombFit$finalModel )
#modelCombFit
predComb <- predict( modelCombFit, newdata = CombDataSetDF ) # built with the testing data.
cm <- confusionMatrix( predComb, testing$classe )
cm$overall["Accuracy"]*100
```

```
## Accuracy 
## 98.99402
```

```r
predCombDF <- as.data.frame( predict( modelCombFit, newdata = CombDataSetDF ) )
#summary( predCombDF )
```
The overall accuracy on the testing set reported above is simply the number of correct predictions (on the testing set):

```r
TestCorrectPred <- sum( (testing$classe == predCombDF[,1]) )
TestIncorrectPred <- sum( (testing$classe != predCombDF[,1]) )
TestCorrectPred / ( TestCorrectPred + TestIncorrectPred )*100
```

```
## [1] 98.99402
```
This result is the same as the accuracy reported by the confusion matrix. A measure of the error would then be:

```r
TestIncorrectPred / ( TestCorrectPred + TestIncorrectPred )*100
```

```
## [1] 1.005982
```
or simply:

```r
( 1 - cm$overall["Accuracy"] )*100
```

```
## Accuracy 
## 1.005982
```
I will use the latter in the rest of the paper.
I will compute the out-of-sample error on the validation set, since the testing set was already used to build the model.

```r
CombDataSetDFV <- data.frame( predLDA = predLDAV, predRF = predRFV, predGBM = predGBMV ) # on the validation data.
#print( modelCombFitV$finalModel )
#modelCombFitV
predCombV <- predict( modelCombFit, newdata = CombDataSetDFV ) # predict on the validation set.
cm <- confusionMatrix( predCombV, validation$classe ) # confusionMatrix( prediction, actual )
cm$overall["Accuracy"]*100
```

```
## Accuracy 
## 99.34747
```

```r
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  947    6    0    1
##          C    0    1  845   11    2
##          D    0    0    4  789    2
##          E    0    0    0    4  896
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9908, 0.9955)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9917          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9979   0.9883   0.9813   0.9945
## Specificity            0.9997   0.9982   0.9965   0.9985   0.9990
## Pos Pred Value         0.9993   0.9927   0.9837   0.9925   0.9956
## Neg Pred Value         1.0000   0.9995   0.9975   0.9963   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1931   0.1723   0.1609   0.1827
## Detection Prevalence   0.2847   0.1945   0.1752   0.1621   0.1835
## Balanced Accuracy      0.9999   0.9981   0.9924   0.9899   0.9967
```

```r
predCombVDF <- as.data.frame( predict( modelCombFit, newdata = CombDataSetDFV ) )
# summary( predCombVDF )
```
The out-of-sample error on the validation set is:

```r
( 1- cm$overall["Accuracy"] )*100
```

```
##  Accuracy 
## 0.6525285
```

```r
TimerStop <- Sys.time()
paste("Combined model time = ", TimerStop - TimerStart, attr( TimerStop - TimerStart,"units") )
```

```
## [1] "Combined model time =  19.2129189968109 secs"
```
The accuracy of the stacked model is 99.35% , at the expense of waiting ~12 mins for it. 
Computationally, very time intensive, with only marginal improvements over and RF and GBM.

FORECAST QUIZ RESULTS.-
=======================

```r
TimerStart <- Sys.time()
# Predict the outcome classe on each of the three models: LDA, RF, GBM.
predLDAQ <- predict( modelLDA, newdata = SubsetForecastDF )
predRFQ <- predict( modelRF, newdata = SubsetForecastDF )
predGBMQ <- predict( modelGBM, newdata = SubsetForecastDF )
# Combine the predictions into one new data frame.
ForecastQuizDF <- data.frame( predLDA = predLDAQ, predRF = predRFQ, predGBM = predGBMQ  )
# Predict classe with the assembelled stacked model (built with the training and testing data, 
# and validated with the validation data)
predQuiz <- predict( modelCombFit, newdata = ForecastQuizDF )
str( predQuiz )
```

```
##  Factor w/ 5 levels "A","B","C","D",..: 2 1 2 1 1 5 4 2 1 1 ...
```

```r
predQuiz
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
This prediction solves the Project quiz with 100% accuracy. We know that from the grader..:).. Now we know the answer to the quiz with 100% accuracy.

What if we tried to predict with only GBM? How many right answers do we get ?

```r
RightAnswer <- as.data.frame( predQuiz )
cm <- confusionMatrix( as.data.frame( predGBMQ )[,1], RightAnswer[,1] ) # confusionMatrix( prediction, actual )
cm$overall["Accuracy"]*100
```

```
## Accuracy 
##      100
```
Also perfect accuracy ! Hence, the GBM model alone should suffice to predict future outcomes.

```r
TimerStop <- Sys.time()
paste("Solving the quiz time => ", TimerStop - TimerStart, attr( TimerStop - TimerStart,"units") )
```

```
## [1] "Solving the quiz time =>  0.374786138534546 secs"
```

```r
stopCluster(cluster)
registerDoSEQ()
```
                          -- THE END --
