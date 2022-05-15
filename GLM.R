#Fitting a Poisson GLM for the learning data.
glm1 <- glm(NClaims ~ Coverage + Fuel + Use + Fleet + Sex + Ageph + BM + Age_car 
            + Power + region, family = poisson(), data = learn, offset = log(Expo))

summary(glm1)


#Add the fitted mean values for all observations in the learning set to 
#the learning data set.
learn$glm <- glm1$fitted.values

#Using the glm model fitted on the learning data to obtain predictions 
#for the number of claims for the testing data. 
#We add these predictions to the testing data. 
test$glm <- predict(glm1, newdata = test, type = "response")

#Dataframe containing in-sample and out-of-sample losses
devianceloss_values <- data.frame(NULL)

devianceloss_values[1,1] <- Poissondeviance(learn$NClaims,learn$glm)
devianceloss_values[1,2] <- Poissondeviance(test$NClaims, test$glm)
devianceloss_values[2,1] <- square_loss(learn$NClaims,learn$glm)
devianceloss_values[2,2] <- square_loss(test$NClaims, test$glm)
devianceloss_values[3,1] <- weighted_square_loss(learn$NClaims,learn$glm)
devianceloss_values[3,2] <- weighted_square_loss(test$NClaims, test$glm)

names(devianceloss_values) <- c('Learning','Testing')









