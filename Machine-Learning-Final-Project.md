Machine Learning Course Project
===============================

### by Eric Dregne

Project Overview
----------------

The goal of this project is to use machine learning to develop a
predictive model for the manner in which exercise was performed. The
data utilized in this case is the Wearable Computing: Accelerometers’
Data Classification of Body Postures and Movements. The data from this
study, gathered from participants wearing movement sensors on different
parts of the body, provides a wide range of information on specific
movements during different manners of exercise, in this case the class
of the exercise. Through modeling this data we will develop the means to
determine the class of exercise performed from given data inputs.

Data Import and Preliminary Analysis
------------------------------------

Here we will load the Weight Lifting Exercise Dataset from the study.
The basic structure of the data will be assessed and we will also load
the R packages necessary for the project.

    library(caret)
    library(knitr)
    library(randomForest)
    library(rattle)

    training_name <- "pml-training.csv"
    testing_name <- "pml-testing.csv"

    dataset_train <- read.csv(training_name, header = TRUE)
    dataset_test <- read.csv(testing_name, header = TRUE)

    dim(dataset_train)

    ## [1] 19622   160

    unique(dataset_train$classe)

    ## [1] "A" "B" "C" "D" "E"

    sapply(dataset_train, class)

    ##                        X                user_name     raw_timestamp_part_1 
    ##                "integer"              "character"                "integer" 
    ##     raw_timestamp_part_2           cvtd_timestamp               new_window 
    ##                "integer"              "character"              "character" 
    ##               num_window                roll_belt               pitch_belt 
    ##                "integer"                "numeric"                "numeric" 
    ##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
    ##                "numeric"                "integer"              "character" 
    ##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
    ##              "character"              "character"              "character" 
    ##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
    ##              "character"              "character"                "numeric" 
    ##           max_picth_belt             max_yaw_belt            min_roll_belt 
    ##                "integer"              "character"                "numeric" 
    ##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
    ##                "integer"              "character"                "numeric" 
    ##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
    ##                "integer"              "character"                "numeric" 
    ##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
    ##                "numeric"                "numeric"                "numeric" 
    ##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
    ##                "numeric"                "numeric"                "numeric" 
    ##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
    ##                "numeric"                "numeric"                "numeric" 
    ##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
    ##                "numeric"                "numeric"                "numeric" 
    ##             accel_belt_x             accel_belt_y             accel_belt_z 
    ##                "integer"                "integer"                "integer" 
    ##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
    ##                "integer"                "integer"                "integer" 
    ##                 roll_arm                pitch_arm                  yaw_arm 
    ##                "numeric"                "numeric"                "numeric" 
    ##          total_accel_arm            var_accel_arm             avg_roll_arm 
    ##                "integer"                "numeric"                "numeric" 
    ##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
    ##                "numeric"                "numeric"                "numeric" 
    ##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
    ##                "numeric"                "numeric"                "numeric" 
    ##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
    ##                "numeric"                "numeric"                "numeric" 
    ##              gyros_arm_y              gyros_arm_z              accel_arm_x 
    ##                "numeric"                "numeric"                "integer" 
    ##              accel_arm_y              accel_arm_z             magnet_arm_x 
    ##                "integer"                "integer"                "integer" 
    ##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
    ##                "integer"                "integer"              "character" 
    ##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
    ##              "character"              "character"              "character" 
    ##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
    ##              "character"              "character"                "numeric" 
    ##            max_picth_arm              max_yaw_arm             min_roll_arm 
    ##                "numeric"                "integer"                "numeric" 
    ##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
    ##                "numeric"                "integer"                "numeric" 
    ##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
    ##                "numeric"                "integer"                "numeric" 
    ##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
    ##                "numeric"                "numeric"              "character" 
    ##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
    ##              "character"              "character"              "character" 
    ##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
    ##              "character"              "character"                "numeric" 
    ##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
    ##                "numeric"              "character"                "numeric" 
    ##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
    ##                "numeric"              "character"                "numeric" 
    ## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
    ##                "numeric"              "character"                "integer" 
    ##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
    ##                "numeric"                "numeric"                "numeric" 
    ##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
    ##                "numeric"                "numeric"                "numeric" 
    ##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
    ##                "numeric"                "numeric"                "numeric" 
    ##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
    ##                "numeric"                "numeric"                "numeric" 
    ##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
    ##                "numeric"                "integer"                "integer" 
    ##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
    ##                "integer"                "integer"                "integer" 
    ##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
    ##                "numeric"                "numeric"                "numeric" 
    ##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
    ##                "numeric"              "character"              "character" 
    ##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
    ##              "character"              "character"              "character" 
    ##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
    ##              "character"                "numeric"                "numeric" 
    ##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
    ##              "character"                "numeric"                "numeric" 
    ##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
    ##              "character"                "numeric"                "numeric" 
    ##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
    ##              "character"                "integer"                "numeric" 
    ##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
    ##                "numeric"                "numeric"                "numeric" 
    ##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
    ##                "numeric"                "numeric"                "numeric" 
    ##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
    ##                "numeric"                "numeric"                "numeric" 
    ##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
    ##                "numeric"                "numeric"                "numeric" 
    ##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
    ##                "integer"                "integer"                "integer" 
    ##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
    ##                "integer"                "numeric"                "numeric" 
    ##                   classe 
    ##              "character"

Preprocessing
-------------

Here we will do our data partition that will be used to obtain the
training set for our model.

    ## [1] 13737   160

The next step is determining how much of the data is missing. We will
eliminate variables that have no data for 75% of their entries.

    # Identify columns that are mostly NA

    train_na <- sapply(training_set, function(x) sum(is.na(x))>0.75*dim(training_set)[1])
    test_na <- sapply(testing_set, function(x) sum(is.na(x))>0.75*dim(testing_set)[1])

    # Remove columns that will not be used to make the model

    training_vars <- training_set[,train_na==FALSE]
    testing_vars <- testing_set[,test_na==FALSE]

The number of variables in the datasets are now 93. Values that are near
zero will have no statistical value in our eventual predictions so we
will remove these columns as well.

    # Remove values with near zero variance
    training_nzv <- nearZeroVar(training_vars)
    length(training_nzv)

    ## [1] 34

    train_df <- training_vars[,-training_nzv]
    testing_nzv <- nearZeroVar(testing_vars)
    length(testing_nzv)

    ## [1] 34

    test_df <- testing_vars[,-testing_nzv]

Finally, we will remove the descriptive data, participant id, timestamp,
etc, that will not be factored into the predictions.

    train_df <- train_df[,-(1:5)]
    test_df <- test_df[,-(1:5)]

Check for missing data.

    sum(is.na(train_df))

    ## [1] 0

    sum(is.na(test_df))

    ## [1] 0

Identifying Principle Components
--------------------------------

With unnecessary values removed from the data we will now search for
variables that will be highly influential for our model. We’ll get a
quick percentage overview of what portion of each class is in the data.

    classe_percent <- prop.table(table(train_df$classe)) * 100
    cbind(freq=table(train_df$classe), percentage=classe_percent)

    ##   freq percentage
    ## A 3906   28.43416
    ## B 2658   19.34920
    ## C 2396   17.44195
    ## D 2252   16.39368
    ## E 2525   18.38101

Next, we’ll look and see which variables are highly coorelated.

    M <- abs(cor(train_df[,-54]))
    diag(M) <- 0
    which(M > 0.8, arr.ind = T)

    ##                  row col
    ## yaw_belt           4   2
    ## total_accel_belt   5   2
    ## accel_belt_y      10   2
    ## accel_belt_z      11   2
    ## accel_belt_x       9   3
    ## magnet_belt_x     12   3
    ## roll_belt          2   4
    ## roll_belt          2   5
    ## accel_belt_y      10   5
    ## accel_belt_z      11   5
    ## pitch_belt         3   9
    ## magnet_belt_x     12   9
    ## roll_belt          2  10
    ## total_accel_belt   5  10
    ## accel_belt_z      11  10
    ## roll_belt          2  11
    ## total_accel_belt   5  11
    ## accel_belt_y      10  11
    ## pitch_belt         3  12
    ## accel_belt_x       9  12
    ## gyros_arm_y       20  19
    ## gyros_arm_x       19  20
    ## magnet_arm_x      25  22
    ## accel_arm_x       22  25
    ## magnet_arm_z      27  26
    ## magnet_arm_y      26  27
    ## accel_dumbbell_x  35  29
    ## accel_dumbbell_z  37  30
    ## gyros_dumbbell_z  34  32
    ## gyros_forearm_z   47  32
    ## gyros_dumbbell_x  32  34
    ## gyros_forearm_z   47  34
    ## pitch_dumbbell    29  35
    ## yaw_dumbbell      30  37
    ## gyros_forearm_z   47  46
    ## gyros_dumbbell_x  32  47
    ## gyros_dumbbell_z  34  47
    ## gyros_forearm_y   46  47

This initial check doesn’t reveal any one variable that stands out as
more correlated than the other so we cannot eliminate any variables at
this time. Given the large number of variables we will be processing,
the random forest method will be used.

Random Forest
-------------

Using random forest, the class of exercise will be factored and we will
see how many prediction errors are present when using the current data.

    train_rf <- randomForest(factor(classe)~.,data = train_df, proximity=TRUE)

    par(mfrow=c(1,2))
    plot(train_rf, main = "Class Errors")
    varImpPlot(train_rf, main = "Variable Importance")

![](Machine-Learning-Final-Project_files/figure-markdown_strict/forest-1.png)

Plotting the importance of each value, the variable num\_window clearly
has the strongest impact on the accuracy of the model. Nonetheless, no
variable listed as an accuracy impact near zero. For this reason, no
more variables will be removed.

The error rate demonstrated for each class value also indicates there is
no improvement in accuracy for any value at more than 200 trees. We’ll
create our initial model using rpart and 5 fold cross validation while
also visualizing the decision tree below.

    modFit <- train(classe~., method="rpart", data=train_df, trControl=trainControl(method="cv", number=5))

    fancyRpartPlot(modFit$finalModel, sub="")

![](Machine-Learning-Final-Project_files/figure-markdown_strict/unnamed-chunk-1-1.png)

    print(modFit$finalModel)

    ## n= 13737 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
    ##    2) roll_belt< 129.5 12486 8627 A (0.31 0.21 0.19 0.18 0.11)  
    ##      4) pitch_forearm< -33.35 1116   10 A (0.99 0.009 0 0 0) *
    ##      5) pitch_forearm>=-33.35 11370 8617 A (0.24 0.23 0.21 0.2 0.12)  
    ##       10) magnet_dumbbell_y< 439.5 9620 6918 A (0.28 0.18 0.24 0.19 0.1)  
    ##         20) roll_forearm< 123.5 5994 3573 A (0.4 0.19 0.19 0.17 0.054) *
    ##         21) roll_forearm>=123.5 3626 2416 C (0.077 0.17 0.33 0.23 0.18) *
    ##       11) magnet_dumbbell_y>=439.5 1750  843 B (0.029 0.52 0.041 0.22 0.19) *
    ##    3) roll_belt>=129.5 1251   47 E (0.038 0 0 0 0.96) *

    preds <- predict(modFit, newdata = test_df)
    confusionMatrix(preds, factor(test_df$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1498  467  473  427  166
    ##          B   30  379   36  177  137
    ##          C  119  293  517  360  284
    ##          D    0    0    0    0    0
    ##          E   27    0    0    0  495
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4909          
    ##                  95% CI : (0.4781, 0.5038)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3351          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8949   0.3327  0.50390   0.0000  0.45749
    ## Specificity            0.6360   0.9199  0.78267   1.0000  0.99438
    ## Pos Pred Value         0.4942   0.4993  0.32867      NaN  0.94828
    ## Neg Pred Value         0.9383   0.8517  0.88196   0.8362  0.89055
    ## Prevalence             0.2845   0.1935  0.17434   0.1638  0.18386
    ## Detection Rate         0.2545   0.0644  0.08785   0.0000  0.08411
    ## Detection Prevalence   0.5150   0.1290  0.26729   0.0000  0.08870
    ## Balanced Accuracy      0.7654   0.6263  0.64328   0.5000  0.72593

The rpart model generated has very limited accuracy (49%) when run on
the test data. We’ll use random forest to create our second model again
using 5 fold cross validation.

    modFit2 <- train(classe~., data=train_df, method="rf", trControl=trainControl(method="cv", number=5))
    modFit2

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    53 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10991, 10988, 10990, 10990, 10989 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9936670  0.9919889
    ##   27    0.9964334  0.9954884
    ##   53    0.9943948  0.9929094
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 27.

We now have a model with greater than 99% accuracy. We’ll now test the
effectiveness of the model by running it on our test set.

    preds2 <- predict(modFit2, newdata = test_df)
    confusionMatrix(preds2, factor(test_df$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    3    0    0    0
    ##          B    0 1134    3    0    2
    ##          C    0    2 1023    3    0
    ##          D    0    0    0  961   10
    ##          E    0    0    0    0 1070
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9961          
    ##                  95% CI : (0.9941, 0.9975)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9951          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9956   0.9971   0.9969   0.9889
    ## Specificity            0.9993   0.9989   0.9990   0.9980   1.0000
    ## Pos Pred Value         0.9982   0.9956   0.9951   0.9897   1.0000
    ## Neg Pred Value         1.0000   0.9989   0.9994   0.9994   0.9975
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1927   0.1738   0.1633   0.1818
    ## Detection Prevalence   0.2850   0.1935   0.1747   0.1650   0.1818
    ## Balanced Accuracy      0.9996   0.9973   0.9980   0.9974   0.9945

The accuracy of the predictions are exceptional. The positive and
negative prediction rate for the test data is almost 99% accurate across
all class types with an out-of-sample error of just ~0.4% for all of the
predictions.

Testing Quiz
------------

Our final model will now use the test set included for the final quiz.

    preds3 <- predict(modFit2, newdata=dataset_test)
    preds3

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Our model answers the 20 quiz questions with 100% accuracy.

Bibliography
------------

**Wearable Computing: Accelerometers’ Data Classification of Body
Postures and Movements** Ugulino, W.; Cardador, D.; Vega, K.; Velloso,
E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers’ Data
Classification of Body Postures and Movements. Proceedings of 21st
Brazilian Symposium on Artificial Intelligence. Advances in Artificial
Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. ,
pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN
978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6\_6. Cited by 85
(Google Scholar)

**Read more:**
<a href="http://groupware.les.inf.puc-rio.br/work.jsf?p1=10335#ixzz6kDAovb7F" class="uri">http://groupware.les.inf.puc-rio.br/work.jsf?p1=10335#ixzz6kDAovb7F</a>
