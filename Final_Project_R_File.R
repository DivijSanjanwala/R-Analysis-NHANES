############### Data Analysis by Divij Sanjanwala (1005887480) ##############

library(tidyverse)
library(NHANES)
library(olsrr)
library(glmnet)
library(rms)
library(MASS)

## The NHANES$AGE parameter shows that data is selected from people > 17 years
small.nhanes <- na.omit(NHANES[NHANES$SurveyYr=="2011_12"& 
                                 NHANES$Age > 17,c(1,3,4,8:11,13,17,20,21,
                                                   25,46,50,51,52,61)])

small.nhanes <- as.data.frame(small.nhanes %>%group_by(ID) %>% filter(row_number()==1))

nrow(small.nhanes)

## Checking whether there are any ID that was repeated. 
## If not then length(unique(small.nhanes$ID))and nrow(small.nhanes) are same ##

length(unique(small.nhanes$ID))

################################################################################

# Creating training and test set
set.seed(1005887480) ## My student number

# Here, the sample size for the training set is set to 500 observations
train <- small.nhanes[sample(seq_len(nrow(small.nhanes)), size = 500),]

nrow(train) ## Finds the number of rows in the training data-set

length(which(small.nhanes$ID %in% train$ID)) # Length/Size of train data-set

test <- small.nhanes[!small.nhanes$ID %in% train$ID,] # Setting test data obs

nrow(test) # Calculating the number of rows in the test data-set

################################################################################


## Step 1: Checking model (17 variable model) diagnostics

## The full model ##
model.full <- lm(BPSysAve ~ . - ID, data = train)

## Converting the binary variables into 1s and 0s according to their responses

train$SmokeNow <- ifelse(train$SmokeNow == "Yes", 1, 0) ## If yes, then 1
train$Gender <- ifelse(train$Gender == "male", 1, 0) ## If male, then 0
train$SleepTrouble <- ifelse(train$SleepTrouble == "Yes", 1, 0) ## If Yes, then 0
train$PhysActive <- ifelse(train$PhysActive == "Yes", 1, 0) ## If Yes, then 0

## Creating a dummy variable matrix. Here essentially, a matrix corresponding
## each individual column is created for each individual categorical variable
## Through this, we create 3 dummy variables for a global variable having 4 categories.
## As we create a matrix, we have a better shot at understanding the relation of
## each category with the BPSysAve variable. 

## Ahead, we will run a Multiple linear regression with all the categorical variables
## against the BPSysAve variable.

MaritalMatrix <- model.matrix(~train$MaritalStatus, data = train)
DepressedMatrix <- model.matrix(~train$Depressed, data = train)
Race3Matrix <- model.matrix(~train$Race3, data = train)
EducationMatrix <- model.matrix(~train$Education, data = train)
HHIncomeMatrix <- model.matrix(~train$HHIncome, data = train)

## Re-running the model with only categorical variables using variable matrix
## Removing the categorical variables with textual data and rerunning the categorical model
## with model.matrix

model.categorical <- lm(train$BPSysAve ~ . - ID - MaritalStatus - Depressed - Race3 - Education - HHIncome + MaritalMatrix + Race3Matrix + DepressedMatrix + EducationMatrix + HHIncomeMatrix, data = train) ## Running the categorical model

# Warning because we have intercepts of each matrix, please ignore the nessage and continue
summary(model.categorical)
plot(model.categorical) # Plotting the residual vs the fitted values of BPSysAve

# Calculating the ANOVA tables to identify which Matrix ie: variable doesn't impact the variable BPSysAve
anova(model.categorical) 

xtable(anova(model.categorical))


## The p-values of Depressed, Race3, and HHIncome high thus, we realize
## that they don't get along with our model enough. Thus we remove them.

model.full_1 <- lm(BPSysAve ~ . - ID - SleepHrsNight - PhysActive - Depressed - Race3 - HHIncome, data = train)

## Once we've used ANOVA, we will proceed with other diagnostic checking procedures
## These procedures will help us check the normality of residuals, and homoscedasticity assumptions

## Once we have shortened the model with unreasonably weird predictors, 
## we check our model for normality and homoscedasticity assumptions. 

summary(model.full_1)

## Through the Q-Q plot, we understand that the normality assumption is met 
## We know this since the Normal Q-Q plot is approximately similar to the dashed 45 deg line.

qqnorm(rstudent(model.full_1))
qqline(rstudent(model.full_1))
# ols_plot_resid_stud(model.full_2)

plot(model.full_1)

## Using VIF, we will attempt to identify variables that have multicollinearity
## In common words, multicolinearity refers to identifying variables that lead to wider confidence intervals
## that lead us to produce less reliable results due to the effect predictors on one another
## in a model.

vif(model.full_1)

## Such variables can be identified using the Variance Inflation factor,
## As a standard practice, a VIF of higher than 5.

## As identified, the variables Weight, Height, BMI have higher VIF
## Thus due to their collinearity i.e. dependence on other variables for the 
## prediction of BPSysAve, they aren't required in the model to predict BPSysAve.

model.full_2 <- lm(train$c ~ . - Weight - Height - BMI - ID - SleepHrsNight 
                   - PhysActive - Depressed - Race3 - HHIncome, data = train)




########################################## Variable Selection #############################################

#Since the normality assumption is met, we can apply the shrinkage methods further to
#Eliminate variables that  significantly impact the model.

categorical_matrix <- cbind(MaritalMatrix, EducationMatrix, train$Gender, train$Age, train$SmokeNow, train$Poverty, train$SleepTrouble)
colnames(categorical_matrix)[16] <- "SleepTrouble"
colnames(categorical_matrix)[15] <- "Poverty"
colnames(categorical_matrix)[14] <- "SmokeNow"
colnames(categorical_matrix)[13] <- "Age"
colnames(categorical_matrix)[12] <- "Gender"

model.full_2.lasso <- glmnet(x = data.matrix(categorical_matrix), 
                             y = train$BPSysAve, standardize = T, alpha = 1)
cv.lasso <- cv.glmnet(x = data.matrix(categorical_matrix), y = I(sqrt(train$BPSysAve)), 
                      standardize = T, alpha = 1)

plot(cv.lasso)
best.lambda <- cv.lasso$lambda.1se
best.lambda
co <- coef(cv.lasso, s = "lambda.1se")

#Selection of the significant features(predictors)

## threshold for variable selection ##

thresh <- 0.00
# select variables #
inds <- which(abs(co) > thresh )
variables <- row.names(co)[inds]
sel.var.lasso<-variables[!(variables %in% '(Intercept)')]
sel.var.lasso


## Running LASSO eliminates many categorical variables inside of Education and
## Marital Status. LASSO identifies a model such that categories inside Marital Status
## and education do not exist in the LASSO model.

## We cross-verify this method using other techniques of AIC and BIC,
## Where we cross check the model we use in the above scenario. 



## We further apply variable selections techniques.

## Based on AIC ##
reduced.model._2 <- lm(train$BPSysAve ~ . - Education - MaritalStatus - Weight - Height - BMI - ID - SleepHrsNight - PhysActive - 
                         Depressed - Race3 - HHIncome, data = train) ## Our model without LASSO removed predictors
summary(reduced.model._2)  
n <- nrow(train)
sel.var.aic <- step(reduced.model._2, trace = 0, k = 2, direction = "both") 
sel.var.aic<-attr(terms(sel.var.aic), "term.labels")   
sel.var.aic

## Based on AIC, we remain with variables - Gender, Age, Poverty, & SleepTrouble. 

## Based on BIC ##
summary(reduced.model._2)  
n <- nrow(train)
sel.var.bic <- step(reduced.model._2, trace = 0, k = log(n), direction = "both") 
sel.var.bic<-attr(terms(sel.var.bic), "term.labels")   
sel.var.bic

## Based on BIC, we remain with variables - Gender & Age.

#################################### End of Variable Selection ##########################################


######################################## Cross - Validation ###############################################

sel.var.aic<-attr(terms(sel.var.aic), "term.labels")   

ols.aic <- ols(train$BPSysAve ~ Gender + Age + Poverty + SleepTrouble, 
               data = train[,which(colnames(train) %in% c(sel.var.aic, "BPSysAve"))], 
               x=T, y=T, model = T)

aic.cross <- calibrate(ols.aic, method = "crossvalidation", B = 10)
## Calibration plot ##
plot(aic.cross, las = 1, xlab = "Predicted BPSysAve", main = "Cross-Validation calibration with AIC")
dev.off()



cv.lasso <- cv.glmnet(x = data.matrix(categorical_matrix), y = I(sqrt(train$BPSysAve)), 
                      standardize = T, alpha = 1)

plot(cv.lasso)
best.lambda <- cv.lasso$lambda.1se
best.lambda
co <- coef(cv.lasso, s = "lambda.1se")

#Selection of the significant features(predictors)

## threshold for variable selection ##

thresh <- 0.00
# select variables #
inds <- which(abs(co) > thresh )
variables <- row.names(co)[inds]
sel.var.lasso<-variables[!(variables %in% '(Intercept)')]
sel.var.lasso


ols.lasso <- ols(train$BPSysAve ~ Age, data = train[,which(colnames(train) %in% c(sel.var.lasso, "BPSysAve"))], 
                 x=T, y=T, model = T)

lasso.cross <- calibrate(ols.lasso, method = "crossvalidation", B = 10)
## Calibration plot ##
plot(lasso.cross, las = 1, xlab = "Predicted BPSysAve", main = "Cross-Validation calibration with LASSO")
dev.off()

## Since AIC gives us a closer approximation with the IDEAL line, we will mode ahead with the variables
## that AIC variable selection gave us. 
## "Gender" "Age" "Poverty" "SleepTrouble"

## LASSO gave us only Age as the variable to be used.

##########################################################################################################

criteria <- function(m){
  n <- length(m$residuals)
  p <- length(m$coefficients) - 1
  RSS <- sum(m$residuals^2)
  R2 <- summary(m)$r.squared
  R2.adj <- summary(m)$adj.r.squared
  AIC <- n*log(RSS/n) + 2*p
  AICc <- AIC + (2*(p+2)*(p+3))/(n-p-1)
  BIC <- n*log(RSS/n) + (p+2)*log(n)
  res <- c(R2, R2.adj, AIC, AICc, BIC)
  names(res) <- c("R Squared", "Adjsuted R Squared", "AIC", "AICc", "BIC")
  return(res)
}

############################################### Model Selection ###########################################

reduced.model._4 <- lm(train$BPSysAve ~ . - SmokeNow - Education - MaritalStatus - 
                         Weight - Height - BMI - ID - SleepHrsNight - 
                         PhysActive - Depressed - Race3 - HHIncome, data = train)

## Model that doesn't include SleepTrouble and Poverty

reduced.model._5 <- lm(train$BPSysAve ~ . - SleepTrouble - Poverty - SmokeNow - Education - MaritalStatus - 
                         Weight - Height - BMI - ID - SleepHrsNight - 
                         PhysActive - Depressed - Race3 - HHIncome, data = train)

criteria(reduced.model._5)

## Both have a similar AIC and BIC, thus removing Poverty and SleepTrouble 
## doesn't impact the model as much.

## We keep Gender as a variable since one of our variable selection techniques
## i.e. BIC returned Gender as a variable to be used. 


######################################### End of  Model Selection #########################################

## We run the criteria function on our final model again, that includes SmokeNow and on the one that 
## doesn't include SmokeNow.

reduced.model._5 <- lm(train$BPSysAve ~ . - SleepTrouble - SmokeNow - Poverty - Education - MaritalStatus - 
                         Weight - Height - BMI - ID - SleepHrsNight - 
                         PhysActive - Depressed - Race3 - HHIncome, data = train)

reduced.model._6 <- lm(train$BPSysAve ~ . - SleepTrouble - Poverty - Education - MaritalStatus - 
                         Weight - Height - BMI - ID - SleepHrsNight - 
                         PhysActive - Depressed - Race3 - HHIncome, data = train)

criteria(reduced.model._5)
criteria(reduced.model._6)

## Since both AIC and BIC are approximately the same, we include SmokeNow variable
## in our final model. ## We will stick to having SmokeNow variable in our model.


########################################################################################################

plot(reduced.model._6)

## The hat values ###
h <- hatvalues(reduced.model._6)
thresh <- 2 * (dim(model.matrix(reduced.model._6))[2])/nrow(train)
w <- which(h > thresh)

## Finding Influential Observations through Cook's distance
D <- cooks.distance(reduced.model._6)
which(abs(D) > qf(0.1, ncol(train), nrow(train) - ncol(train))) ## qf(percentile, p parameter, n - p df)
# Running cooks distance model, we find no observations influential according
# to the threshold set by the cooks distance formula

######### Run only till here since the below code removes observations from the dataset ###########

## DFFITS
dfits <- dffits(reduced.model._6)
remove <- which(abs(dfits) > 2*sqrt(5/nrow(train)))

## UNCOMMENT THE BELOW LINE TO REMOVE THE INFLUENTIAL OBSERVATIONS

# train <- train[-c(315, 38, 470, 251, 385, 669, 91, 118, 
#                   664, 688, 513, 571, 687, 161, 332, 723, 437, 471),]

nrow(train) ## Run the above line then this line to verify that the 11 influential observations are removed.

## DFBETAS
dfb <- dfbetas(reduced.model._6)

which(abs(dfb[,1]) > 2/sqrt(nrow(train)))

## When we run DFBETAS and DFFITS, we find out that the influential points that
## the DFFITS function reveals are the subset of the influential points the
## DFBETAS reveals. Thus, we remove the influential values DFFITS from the dataset.

summary(reduced.model._4)

beta_0 <- coef(reduced.model._6)["(Intercept)"][1]
beta_1 <- coef(reduced.model._6)["Gender"][1]
beta_2 <- coef(reduced.model._6)["Age"][1]
beta_3 <- coef(reduced.model._6)["SmokeNow"][1]

test$SmokeNow <- ifelse(test$SmokeNow == "Yes", 1, 0) ## If yes, then 1
test$Gender <- ifelse(test$Gender == "male", 1, 0) ## If male, then 0

## Given the coefficient of SmokeNow as -1.423056, we know that it has a significant
## impact on the model and is an integral part of the model.

yhat_BPSysAve <- 100.725  + (5.602392  * test$Gender) + 
  (0.4150575 * test$Age) + (-1.423056 * test$SmokeNow)

mean((test$BPSysAve - yhat_BPSysAve)^2)








