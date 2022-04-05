glm1 <- glm(NClaims ~ Coverage + Fuel + Use + Fleet + Sex + Ageph + BM + Age_car 
            + Power + region, family = poisson(), data = learn, offset = log(Expo))
summary(glm1)

glm1$fitted.values
max(glm1$fitted.values)
min(glm1$fitted.values)

glm3 <- glm(NClaims ~ Coverage + Fuel + Use + Fleet + Sex + Ageph + BM + Age_car 
            + Power + region, family = poisson(), data = learn2, offset = log(Expo))
summary(glm3)
min(glm3$fitted.values)

#Observation: the claim data was bimodal -> is using a glm for this possible? 
#probably not since no of the models would give a good fit for this 
learn_claims <- learn %>% filter(learn$Claim > 0)
glm2 <- glm(Claim ~ Coverage + Fuel + Use + Fleet + Sex + Ageph + BM + Age_car 
            + Power + region , family = Gamma(link = "inverse"), data = learn_claims)
summary(glm2)




