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

names(devianceloss_values) <- c('Learning','Testing')

#Variable selection based on the likelihood ratio test
drop1(glm1, test = "LRT")

#p-value for the model without respectively Sex and Age_car is > 0.05 so we fit GLM without these features 

#Fitting a Poisson GLM for the learning data.
glm2 <- glm(NClaims ~ Coverage + Fuel + Use + Fleet  + Ageph + BM
            + Power + region, family = poisson(), data = learn, offset = log(Expo))

summary(glm2)

#Add the fitted mean values for all observations in the learning set to 
#the learning data set.
learn$glm2 <- glm2$fitted.values

#Using the glm model fitted on the learning data to obtain predictions 
#for the number of claims for the testing data. 
#We add these predictions to the testing data. 
test$glm2 <- predict(glm2, newdata = test, type = "response")

devianceloss_values[1,3] <- Poissondeviance(learn$NClaims,learn$glm2)
devianceloss_values[1,4] <- Poissondeviance(test$NClaims, test$glm2)
devianceloss_values[2,3] <- square_loss(learn$NClaims,learn$glm2)
devianceloss_values[2,4] <- square_loss(test$NClaims, test$glm2)

#Plot the estimated frequency for the test data (for each feature)
p1 <- plot_freq(test, "Coverage", "frequency by coverage", "GLM", "glm")
p2 <- plot_freq(test, "region", "frequency by region", "GLM", "glm")
p3 <- plot_freq(test, "Fuel", "frequency by fuel", "GLM", "glm")
p4 <- plot_freq(test, "Use", "frequency by use", "GLM", "glm")
grid.arrange(p1,p2,p3,p4)

p5 <- plot_freq(test, "Power", "frequency by power", "GLM", "glm")
p6 <- plot_freq(test, "Ageph", "frequency by age", "GLM", "glm")
p7 <- plot_freq(test, "Age_car", "frequency by age of car", "GLM", "glm")
p8 <- plot_freq(test, "BM", "frequency by BM", "GLM", "glm")
grid.arrange(p5,p6,p7,p8)

p9 <- plot_freq(test, "Sex", "frequency by sex", "GLM", "glm")
p10 <- plot_freq(test, "Fleet", "frequency by fleet", "GLM", "glm")
grid.arrange(p9,p10)







