#Setting the directory
dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dir)
KULbg <- "#116E8A"

#Installing tensorflow package
install.packages("tensorflow")
tensorflow::install_tensorflow(version = "2.6.2")

#Loading the libraries
library(keras)
library(tensorflow)
library(locfit)
library(magrittr)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)
library(gridExtra)
library(tidyr)
library(corrplot)
library(OpenML)
library(farff)
library(rgdal)
library(splitTools)
RNGversion("3.5.0")

#Setting the seed such that this is fixed whenever the 
#code is run 
seed <- 100
Sys.setenv(PYTHONHASHSEED = seed)
set.seed(seed)
reticulate::py_set_seed(seed)
tensorflow::tf$random$set_seed(seed)
tensorflow::tf$compat$v1$disable_eager_execution()
############# Adapting the data ########## ################

#Upload the data
data <- read.delim('PC_data.txt', header = TRUE, stringsAsFactors = FALSE)
summary(data)

#Rename the columns 
names0 <- c("Id","NClaims","Claim","Avg_Claim","Expo","Coverage","Fuel","Use","Fleet","Sex","Ageph","BM","Age_car","Power","PC","Town","Long","Lat")
names(data) <- names0

#Convert type from character to factor 
data$Coverage <- as.factor(data$Coverage)
data$Fuel <- as.factor(data$Fuel)
data$Use <- as.factor(data$Use)
data$Fleet <- as.factor(data$Fleet)
data$Sex <- as.factor(data$Sex)

#Making sure that number of claims is integer 
data$NClaims <- as.integer(data$NClaims)

#Converting the numeric values for the claims into integer values 
data$Claim <- as.integer(data$Claim)
data$Avg_Claim <- as.integer(data$Avg_Claim)

#Adding new variable indicating the region based on the postcode.
#This variable will simplify the further process.
PC_breaks <- c(1000,1300,1500,2000,3000,3500,4000,5000,6000,6600,7000,8000,9000,10000)

data <- data %>% mutate(region = cut(PC,breaks = PC_breaks, labels = c('Brussels','Walloon Brabant','Flemish Brabant','Antwerp','Flemish Brabant','Limburg','Liege','Namur','Hainaut','Luxembourg','Hainaut','West Flanders', 'East Flanders'),right = FALSE))

#Adding new variable binning the exposure to simplify the plots. 
Expo_breaks <- c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)

data <- data %>% mutate(Expo_bin = cut(Expo,breaks = Expo_breaks, labels = c('1','2','3','4','5','6','7','8','9', '10')))
                                
#Adding the frequency 
data <- data %>% mutate(freq = NClaims/Expo)

#Analysing the dataset
str(data)
summary(data)
data %>% arrange(desc(freq))

#Capping some variables to avoid distortion of the analysis. 
#Motivation for the capping is provided below, when analysing 
#the variables. 
#After the capping, the dataset is ordered decreasing in function 
#of the average claim amounts. 
data_cap <- data %>% mutate(Ageph = pmin(Ageph,88), BM = pmin(BM,18),
                            Age_car = pmin(Age_car,25), Power = pmax(13,pmin(Power,125)))

#Two columns are added to the dataset. 
#RandNN contains standard normally distributed random numbers
#RandUN contains random numbers generated from a uniform distribution 
#on the interval from - sqrt3 to sqrt3. 
#Both vectors are normalized

set.seed(seed)

data_cap <- data_cap %>% mutate(
  RandNN = rnorm(nrow(data_cap)),
  RandNN = scale_no_attr(RandNN),
  RandUN = runif(nrow(data_cap), min = -sqrt(3), max = sqrt(3)),
  RandUN = scale_no_attr(RandUN)
)

############ First inspection of the data #################

############## 1. Age ################

#Age ranges from 18 up to 95
range(data$Ageph)

#Plotting histogram in function of age policyholders for
#number of policies. There are very little policies 
#for young and old drivers. 
number_policies_Age <- ggplot(data) + geom_bar(aes(x = Ageph), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Age of policyholder', y = 'Number of policies') + ggtitle('Policies per age category')

sum(data$Ageph < 20)/nrow(data)
sum(data$Ageph > 88)/nrow(data)

#Histogram for the total exposure. Histogram looks like 
#the histogram for the number of policies, which is intuitive.
#As the number of policies for an age group increases
#it is intuitive that the exposure will increase. 
#Furthermore, most policyholders take out policies with 
#exposure of one year.
expo_age <- ggplot(data %>% group_by(Ageph) %>% summarise(totalexpo = sum(Expo)), aes(x = Ageph, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of policyholder', y = 'Total exposure') + ggtitle('The total exposure per age category')

grid.arrange(number_policies_Age, expo_age, ncol = 2)

#Plotting histogram for the number of claims per age policyholder. 
#Again, there are very little claims filed by young and old 
#drivers compared to the other drivers. This is intuitive 
#since we also do not have a lot of policies for these 
#drivers.
nclaims_age <- ggplot(data %>% group_by(Ageph) %>% summarise(totalclaims = sum(NClaims)), aes(x = Ageph, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of policyholder', y = 'Total number of claims') + ggtitle('The total number of claims per age category')
nclaims_age

#Empirical frequency per age policyholder. We observe 
#a very high frequency for drivers of age 18. These young drivers 
#have an empirical frequency of 1.08. 

freq_age <- ggplot(data %>% group_by(Ageph) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Ageph, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of policyholder', y = 'Empirical frequency') + ggtitle('The empirical frequency per age category')

#Looking at the data it is clear that these drivers have 
#rather short exposure whilst regularly filing claims. 
#As a result, a high frequency is obtained.
#Since these drivers are young, it is intuitive that 
#they would indeed regularly have an accident. 
#We note that the dataset only contains 16 policies 
#for drivers of age 18, which is a very small number.
data %>% filter(data$Ageph == 18)

freq_age_col <- data %>% group_by(Ageph) %>% summarize(Freq = sum(NClaims) / sum(Expo))

sum(data$Ageph == 18)

#On the other hand, for drivers older than 92 the empirical 
#frequency is zero. Again, not many policies were obtained 
#for these drivers but having zero accidents seem strange 
#since we expect older drivers to indeed be less safe 
#than slightly younger drivers. From age 89 up to 95 
#we observe a very low empirical frequency. Therefore
#we will cap the age at age 88 -> making sure 
#that this does not distort the model. 

#The empirical severity given that there is a claim 
#for the policyholders per age category. 
#Clearly, since for policyholder with age >92 there are no claims 
#the value for severity will be NaN.
sev_age <- ggplot(data %>% group_by(Ageph) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Ageph, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of policyholder', y = 'Empirical severity') + ggtitle('The empirical severity per age category')

grid.arrange(freq_age, sev_age, ncol = 2)


################ 2. Coverage ##############

#Most drivers have TPL coverage, about 60% of the policyholders. 
summary(data$Coverage)
95136/nrow(data)

#Number of policies per coverage type. 
number_policies_Coverage <- ggplot(data) + geom_bar(aes(x = Coverage), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Type of coverage', y = 'Number of policies') + ggtitle('Policies per coverage type')

number_policies_Coverage

#Total exposure per coverage type.
#This again looks similar to histogram for number of policies.
expo_coverage <- ggplot(data %>% group_by(Coverage) %>% summarise(totalexpo = sum(Expo)), aes(x = Coverage, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Type of coverage', y = 'Total exposure') + ggtitle('The total exposure per coverage type')

grid.arrange(number_policies_Coverage, expo_coverage,ncol = 2)

#Number of claims per coverage type
nclaims_coverage <- ggplot(data %>% group_by(Coverage) %>% summarise(totalclaims = sum(NClaims)), aes(x = Coverage, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Type of coverage', y = 'Total number of claims') + ggtitle('The total number of claims per coverage type')

nclaims_coverage

#Empirical frequency per coverage type. 
#It is visible that the empirical frequency does not differ a lot between 
#the different coverage types. Therefore, we would expect that this variable 
#does not have a big impact on the frequency of an accident.
freq_coverage <- ggplot(data %>% group_by(Coverage) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Coverage, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Type of coverage', y = 'Empirical frequency') + ggtitle('The empirical frequency per coverage type')

freq_coverage

#Empirical severity per coverage type. Clearly, we expect the severity 
#to be slightly less for drivers with PO coverage.
sev_coverage <- ggplot(data %>% group_by(Coverage) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Coverage, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Type of coverage', y = 'Empirical severity') + ggtitle('The empirical severity per coverage type')

sev_coverage


################# 3. Fuel #############
#Most drivers use gasoline as fuel, about 69% of the policyholders.
summary(data$Fuel)
summary(data$Fuel)[2]/nrow(data)


#Number of policies per fuel type. 
number_policies_fuel <- ggplot(data) + geom_bar(aes(x = Fuel), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Type of fuel', y = 'Number of policies') + ggtitle('Policies per fuel type')

number_policies_fuel

#Total exposure per fuel type.
#This again looks similar to histogram for number of policies.
expo_fuel <- ggplot(data %>% group_by(Fuel) %>% summarise(totalexpo = sum(Expo)), aes(x = Fuel, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Type of fuel', y = 'Total exposure') + ggtitle('The total exposure per fuel type')

#Number of claims per fuel type
nclaims_fuel <- ggplot(data %>% group_by(Fuel) %>% summarise(totalclaims = sum(NClaims)), aes(x = Fuel, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Type of fuel', y = 'Total number of claims') + ggtitle('The total number of claims per fuel type')

nclaims_fuel

#Empirical frequency per fuel type. 
#The empirical frequency is higher for diesel than for gasoline users. 
freq_fuel <- ggplot(data %>% group_by(Fuel) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Fuel, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Type of fuel', y = 'Empirical frequency') + ggtitle('The empirical frequency per fuel type')

freq_fuel

#Empirical severity per fuel type. 
#Whilst the empirical frequency is higher for diesel than for gasoline motors, 
#it is visible that the empirical severity is higher for gasoline users. 
sev_fuel <- ggplot(data %>% group_by(Fuel) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Fuel, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Type of fuel', y = 'Empirical severity') + ggtitle('The empirical severity per fuel type')

sev_fuel

#In general we expect diesel users to have more accidents at a lower cost
#whilst gasoline users have less accidents, but these will on average, come at a higher cost.


################# 4. Use ######################
#Most drivers use the car for private use, about 95% of the policyholders. 
summary(data$Use)
summary(data$Use)[1]/nrow(data)

#Number of policies per usage type. 
number_policies_use <- ggplot(data) + geom_bar(aes(x = Use), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Use of car', y = 'Number of policies') + ggtitle('Policies per usage')

number_policies_use

#Total exposure per usage type.
#This again looks similar to histogram for number of policies.
expo_use <- ggplot(data %>% group_by(Use) %>% summarise(totalexpo = sum(Expo)), aes(x = Use, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Use of car', y = 'Total exposure') + ggtitle('The total exposure per usage')

expo_use

#Number of claims per usage type. 
#We observe more claims for private use than for work use. 
#Of course, this is intuitive since we also have more policies for these policyholders 
#Therefore, observing frequency would give us a more comparable approach, 
#since this offers us a relative view.
nclaims_use <- ggplot(data %>% group_by(Use) %>% summarise(totalclaims = sum(NClaims)), aes(x = Use, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Use of car', y = 'Total number of claims') + ggtitle('The total number of claims per usage')

nclaims_use

#Empirical frequency per usage type. 
#It is visible that the empirical frequency does not differ a lot between 
#the different usage types. Therefore, we expect that this variable 
#does not have a big impact on the frequency of an accident.
freq_use <- ggplot(data %>% group_by(Use) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Use, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Use of car', y = 'Empirical frequency') + ggtitle('The empirical frequency per usage')

freq_use

#Empirical severity per usage type. We expect the severity for policyholders 
#using their car for private reasons to be slightly more than for work use. 
#It is visible that the severity does not differ drastically between the two 
#usage types. 
sev_use <- ggplot(data %>% group_by(Use) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Use, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Use of car', y = 'Empirical severity') + ggtitle('The empirical severity per usage')

sev_use

#We expect the use of the car to not have a big impact on both the 
#frequency as well as severity of the claims.

################### 5. Fleet ##################
#Most drivers do not have a fleet insurance, about 96.8% of the policyholders. 
summary(data$Fleet)
summary(data$Fleet)[1]/nrow(data)

#Number of policies for fleet. 
number_policies_fleet <- ggplot(data) + geom_bar(aes(x = Fleet), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Fleet', y = 'Number of policies') + ggtitle('Policies for fleet')

number_policies_fleet

#Total exposure for fleet.
#This again looks similar to histogram for number of policies.
expo_fleet <- ggplot(data %>% group_by(Fleet) %>% summarise(totalexpo = sum(Expo)), aes(x = Fleet, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Fleet', y = 'Total exposure') + ggtitle('The total exposure for fleet')

expo_fleet

#Number of claims for fleet. Obviously, since most policies do not cover 
#for a fleet also most claims will occur for the policyholders that 
#are not part of a fleet.
nclaims_fleet <- ggplot(data %>% group_by(Fleet) %>% summarise(totalclaims = sum(NClaims)), aes(x = Fleet, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Fleet', y = 'Total number of claims') + ggtitle('The total number of claims for fleet')

nclaims_fleet

#Empirical frequency for fleet. 
#It is visible that the empirical frequency is a bit higher for 
#drivers who do not hold a fleet insurance. The difference 
#between the empirical frequency of drivers that are part of a fleet 
#and the ones who are not, is not significant.

freq_fleet <- ggplot(data %>% group_by(Fleet) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Fleet, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Fleet', y = 'Empirical frequency') + ggtitle('The empirical frequency for fleet')

freq_fleet

#Empirical severity for fleet. We expect that the severity will be higher 
#for drivers who do not have a fleet insurance than for the ones 
#who do have this.
sev_fleet <- ggplot(data %>% group_by(Fleet) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Fleet, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Fleet', y = 'Empirical severity') + ggtitle('The empirical severity for fleet')

sev_fleet

#We expect policyholders who do not hold a fleet insurance to have 
#more frequent accidents and more severe accidents.


################ 6. Sex ####################
#Most policyholders are male, about 73.5% of the policyholders. 
summary(data$Sex)
summary(data$Sex)[2]/nrow(data)

#Number of policies per sex. 
number_policies_sex <- ggplot(data) + geom_bar(aes(x = Sex), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Sex', y = 'Number of policies') + ggtitle('Policies per sex')

number_policies_sex

#Total exposure per sex.
#This again looks similar to histogram for number of policies.
expo_sex <- ggplot(data %>% group_by(Sex) %>% summarise(totalexpo = sum(Expo)), aes(x = Sex, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Sex', y = 'Total exposure') + ggtitle('The total exposure per sex')

expo_sex

#Number of claims per sex. Again, the number of claims for males is larger 
#than the number of claims for females as is also the case for the number 
#of policies.
nclaims_sex <- ggplot(data %>% group_by(Sex) %>% summarise(totalclaims = sum(NClaims)), aes(x = Sex, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Sex', y = 'Total number of claims') + ggtitle('The total number of claims per sex')

nclaims_sex

#Empirical frequency per sex. 
#It is visible that the empirical frequency does not differ a lot between 
#the different sexes. Therefore, we expect that the sex does not have 
#a significant impact on the frequency of car accidents.
freq_sex <- ggplot(data %>% group_by(Sex) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Sex, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Sex', y = 'Empirical frequency') + ggtitle('The empirical frequency per sex')

freq_sex

#Empirical severity per sex. The empirical severity for females 
#is slightly higher than for males. It is visible that the difference 
#between them is not significant. Therefore, we expect that the 
#sex of a driver does not have a large impact on the severity of an accident.
sev_sex <- ggplot(data %>% group_by(Sex) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Sex, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Sex', y = 'Empirical severity') + ggtitle('The empirical severity per sex')

sev_sex



################ 7. Bonus Malus ############
#The Bonus-malus level ranges from 0 to 22. 
range(data$BM)

#Number of policies per bonus malus level.
#Clearly, there are a lot of policies with bonus-malus level equal to zero. 
#Further, we observe very few policies with a bonus-malus level higher than 15.
#This is rather good news for an insurance company since the portfolio 
#contains a lot of very good drivers.
number_policies_BM <- ggplot(data) + geom_bar(aes(x = BM), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Bonus-malus level', y = 'Number of policies') + ggtitle('Policies per bonus-malus level')

sum(data$BM > 18)/nrow(data)

#Total exposure per bonus malus level.
#This again looks similar to histogram for number of policies.
expo_BM <- ggplot(data %>% group_by(BM) %>% summarise(totalexpo = sum(Expo)), aes(x = BM, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Bonus-malus level', y = 'Total exposure') + ggtitle('The total exposure per bonus-malus level')

expo_BM_list <- data %>% group_by(BM) %>% summarise(totalexpo = sum(Expo))

grid.arrange(number_policies_BM, expo_BM, ncol = 2)

#Number of claims per bonus malus level
nclaims_BM <- ggplot(data %>% group_by(BM) %>% summarise(totalclaims = sum(NClaims)), aes(x = BM, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Bonus-malus level', y = 'Total number of claims') + ggtitle('The total number of claims per bonus-malus level')

nclaims_BM

#Empirical frequency per bonus malus level. 
#It is clear that the frequency is increasing as the bonus-malus level increases, 
#which is intuitive since a higher BM level indicates a rather bad driver.
#Consequentially, the frequency will be higher for these drivers. 
#The observations for the frequency histogram corresponding to the 
#censored data remain similar.

freq_BM <- ggplot(data %>% group_by(BM) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = BM, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Bonus-malus level', y = 'Empirical frequency') + ggtitle('The empirical frequency per BM level')

freq_BM_cap <- ggplot(data_cap %>% group_by(BM) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = BM, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Bonus-malus level', y = 'Empirical frequency') + ggtitle('The empirical frequency per BM level - capped')

grid.arrange(freq_BM, freq_BM_cap, ncol = 2)

#Empirical severity per bonus-malus level. Clearly, for drivers with 
#bonus malus level equal to 22 the empirical severity is extremely high.
#It is clear that we only have 32 policies for this BM level. 
#Analysing these drivers, it is clear that there is one driver with 
#a claim amount of 407477 resulting into an extremely high severity 
#for this group of drivers. To not distort the data analysis by 
#this one extreme observation, we right-censored the data for BM at 18.
sev_BM <- ggplot(data %>% group_by(BM) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = BM, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Bonus-malus level', y = 'Empirical severity') + ggtitle('The empirical severity per BM level')

sev_BM_cap <- ggplot(data_cap %>% group_by(BM) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = BM, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Bonus-malus level', y = 'Empirical severity') + ggtitle('The empirical severity per BM level - capped')

sum(data$BM == 22)
 
grid.arrange(sev_BM, sev_BM_cap, ncol = 2)

########## 8. Age car ###############
range(data$Age_car)
summary(data$Age_car)

#Number of policies per car age. 
#Very few policies for cars that are older than 20 years, only 766. 
#This attributes to approx. 0.4% of the total number of policies.
#Policies for cars older than 25 years = 281, only 0.17% of the total 
#number of policies available.
number_policies_AgeCar <- ggplot(data) + geom_bar(aes(x = Age_car), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Age of car', y = 'Number of policies') + ggtitle('Policies per car age')

number_policies_AgeCar

number_policies_AgeCar_datcap <- ggplot(data_cap) + geom_bar(aes(x = Age_car), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Age of car', y = 'Number of policies') + ggtitle('Policies per car age - censored')

number_policies_AgeCar_datcap
sum(data$Age_car == 0)/nrow(data)
sum(data$Age_car > 25)
sum(data$Age_car > 25)/nrow(data)

#Total exposure per car age.
#This again looks similar to histogram for number of policies.
expo_AgeCar <- ggplot(data %>% group_by(Age_car) %>% summarise(totalexpo = sum(Expo)), aes(x = Age_car, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of car', y = 'Total exposure') + ggtitle('The total exposure per car age')

expo_AgeCar

grid.arrange(number_policies_AgeCar, expo_AgeCar, ncol = 2)


#Number of claims per car age
nclaims_AgeCar <- ggplot(data %>% group_by(Age_car) %>% summarise(totalclaims = sum(NClaims)), aes(x = Age_car, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of car', y = 'Total number of claims') + ggtitle('The total number of claims per car age')

nclaims_AgeCar

#Empirical frequency per car age. Observing the empirical frequency, we would 
#expect that for cars older than 32 there are no accidents. 
#For the cars aged 37 we observed one claim, resulting in a frequency. 
#Since we do not have a lot of observations for cars older than 25 
#we will right-censor the data at this point to avoid distortion of 
#the analysis 
freq_AgeCar <- ggplot(data %>% group_by(Age_car) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Age_car, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of car', y = 'Empirical frequency') + ggtitle('The empirical frequency per car age')

freq_AgeCar_col <- data %>% group_by(Age_car) %>% summarize(Freq = sum(NClaims) / sum(Expo))

data %>% filter(Age_car == 37)
sum(data$Age_car == 37)

sum(data$Age_car > 25)

freq_AgeCar_cap <- ggplot(data_cap %>% group_by(Age_car) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Age_car, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of car', y = 'Empirical frequency') + ggtitle('The empirical frequency per car age - capped')

grid.arrange(freq_AgeCar, freq_AgeCar_cap, ncol = 2)
#Observing the frequency we do not expect the age of the car to have a 
#big impact on the frequency since this seems to be rather equal across 
#the different ages. We note that for new cars (0-1 years old), we expect the frequency 
#to be a little bit higher than for older cars.

#Empirical severity per car age. Clearly, looking at the uncensored data 
#we observe an enormous peak for car age = 37. To not distort the data 
#we already put a cap on the data at car age = 25. 
sev_AgeCar <- ggplot(data %>% group_by(Age_car) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Age_car, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of car', y = 'Empirical severity') + ggtitle('The empirical severity per car age')


sev_AgeCar_cap <- ggplot(data_cap %>% group_by(Age_car) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Age_car, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of car', y = 'Empirical severity') + ggtitle('The empirical severity per car age - capped')

grid.arrange(sev_AgeCar, sev_AgeCar_cap, ncol = 2)

#The severity seems to vary across the different ages with no significant
#outliers for the censored data. 

################# 9. Power #####################
#It is clear that the data is right-skewed when observing this in function 
#of the power of the car.
summary(data$Power)

#Number of policies per power. 
number_policies_Power <- ggplot(data) + geom_bar(aes(x = Power), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Power of car', y = 'Number of policies') + ggtitle('Policies per power')

number_policies_Power
sum(data$Power > 125)

#Total exposure per power.
#This again looks similar to histogram for number of policies.
expo_Power <- ggplot(data %>% group_by(Power) %>% summarise(totalexpo = sum(Expo)), aes(x = Power, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Power of car', y = 'Total exposure') + ggtitle('The total exposure per power')

grid.arrange(number_policies_Power, expo_Power, ncol = 2)

#Number of claims per power.
nclaims_Power <- ggplot(data %>% group_by(Power) %>% summarise(totalclaims = sum(NClaims)), aes(x = Power, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Power of car', y = 'Total number of claims') + ggtitle('The total number of claims per power')

nclaims_Power

#Empirical frequency per power. 
#It is visible that the frequency remains stable up to power = 150. 
#From this point on, the empirical frequency is significantly higher 
#than for cars with lower power. This seems intuitive since cars with 
#higher power will probably go faster and we would expect these drivers 
#to have a more agessive driving style. Thus resulting into more accidents.
#For cars with power = 152 we observe an enormous peak. Resulting from 
#a driver filing 3 accidents with a very short exposure. 
#To avoid distortion of the analysis by this one observation, we will 
#right-censor the data at Power = 125.

#Furthermore, we note that there is only one policyholder with power = 10 
#resulting in a rather high frequency. To avoid distortion, we 
#left-censor the data at power = 13. 
freq_Power <- ggplot(data %>% group_by(Power) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Power, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Power of car', y = 'Empirical frequency') + ggtitle('The empirical frequency per power')

data %>% filter(Power == 10)
data %>% filter(Power == 152)

data_power_count <- data %>% group_by(Power) %>% summarise(n = n())
sum(data$Power > 125)/nrow(data)

freq_Power_cap <- ggplot(data_cap %>% group_by(Power) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = Power, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Power of car', y = 'Empirical frequency') + ggtitle('The empirical frequency per power - capped')

grid.arrange(freq_Power, freq_Power_cap, ncol = 2)

#For the censored data we observe an increasing trend in frequency in function 
#of the power of the car, which is intuitive. 

#Empirical severity per power. We observe a number of peaks. For observations 
#with power > 125 there is only one peak. By right censoring the data 
#this peak is not visible anymore, since there are also very small 
#severity amounts for claims with power > 125.
sev_Power <- ggplot(data %>% group_by(Power) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Power, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Power of car', y = 'Empirical severity') + ggtitle('The empirical severity per power')

sev_Power_cap <- ggplot(data_cap %>% group_by(Power) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Power, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Power of car', y = 'Empirical severity') + ggtitle('The empirical severity per power - capped')

grid.arrange(sev_Power, sev_Power_cap, ncol = 2)

sev_Power_column <- data_cap %>% group_by(Power) %>% summarise(sev = sum(Claim)/sum(NClaims)) 

power63 <- data_cap %>% filter(Power == 63)
power115 <- data_cap %>% filter(Power == 115) 

which.max(power63$Claim)
#Observing the claim amount of the policyholders with power 115 it is clear that 
#there is one driver with a claim amount of 200754, resulting in the 
#very large peak. Analysing the policyholder with power = 63, there is 
#one claim of claim amount 1.000.000 attributing to this peak. 

############### 10. Region #####################
#There does not seem to be a significant difference in the number of 
#policyholders per region. 
summary(data$region)

#Number of policies per region. The number of policies seems to be 
#uniformly distributed across the Belgian regions with the small exception 
#of Namen. For this region there seems to be less policies compared to
#the other regions.
number_policies_Region<- ggplot(data) + geom_bar(aes(x = region), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Region', y = 'Number of policies') + ggtitle('Policies per region')

number_policies_Region

#Total exposure per region.
#This again looks similar to histogram for number of policies.
expo_Region <- ggplot(data %>% group_by(region) %>% summarise(totalexpo = sum(Expo)), aes(x = region, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Region', y = 'Total exposure') + ggtitle('The total exposure per region')

expo_Region

#Number of claims per region
nclaims_Region <- ggplot(data %>% group_by(region) %>% summarise(totalclaims = sum(NClaims)), aes(x = region, y = totalclaims)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Region', y = 'Total number of claims') + ggtitle('The total number of claims per region')

nclaims_Region

#Empirical frequency per region. There are no significant differences in 
#the empirical frequency between the region. The amounts seem to be 
#more or less equal. We note that we expect the frequency of accidents 
#around Brussels to be a little higher compared to the other regions. 
freq_Region <- ggplot(data %>% group_by(region) %>% summarize(Freq = sum(NClaims) / sum(Expo)),aes(x = region, y = Freq))+ geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Region', y = 'Empirical frequency') + ggtitle('The empirical frequency per region')

freq_Region

#Empirical severity per region. The empirical severity does not seem 
#to differ a lot across the different regions. 
sev_Region <- ggplot(data %>% group_by(region) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = region, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Region', y = 'Empirical severity') + ggtitle('The empirical severity per region')

sev_Region

#Conclusion: We expect the region to not have a big impact on the 
#severity and frequency of the policyholders. 

################# 11. Spatial view ######################
#This function can be used to read in the shape file of belgium
readShapefile = function(){
  belgium_shape <- readOGR(dsn = path.expand("./shape file Belgie postcodes"), 
                           layer = "npc96_region_Project1")
  belgium_shape <- spTransform(belgium_shape, CRS("+proj=longlat +datum=WGS84"))
  belgium_shape$id <- row.names(belgium_shape)
  return(belgium_shape)
}

##### (i) Exposure 
expo_per_postal_code <- data_cap %>% group_by(PC) %>% summarize(totalexpo = sum(Expo))
range(expo_per_postal_code$totalexpo)
belgium_shape = readShapefile()
belgium_shape@data <- left_join(belgium_shape@data,expo_per_postal_code, by = c('POSTCODE' = 'PC'))

#We divide the total exposure across the municipalities in Belgium 
#in three bins: low, average and high.
belgium_shape@data$expo_class <- cut(belgium_shape@data$totalexpo, breaks = quantile(belgium_shape@data$totalexpo, 
                                                                                c(0,0.2,0.8,1),na.rm = TRUE),right = FALSE, include.lowest = TRUE, labels = c('low','average','high'))
#Mapping the Belgium with the total exposure bins
belgium_shape_f <- fortify(belgium_shape)
belgium_shape_f <- left_join(belgium_shape_f,belgium_shape@data)
plot.eda.map <- ggplot(belgium_shape_f, aes(long,lat, group = group)) +
  geom_polygon(aes(fill = belgium_shape_f$expo_class), colour = 'black', size = 0.1)
plot.eda.map <- plot.eda.map + theme_bw() + labs( fill = 'Exposure') + scale_fill_brewer(palette = 'Blues', na.value = 'White')
plot.eda.map

##### (ii) Frequency 
freq_per_postal_code <- data %>% group_by(PC) %>% summarize(freq = sum(NClaims) / sum(Expo))
belgium_shape = readShapefile()
belgium_shape@data <- left_join(belgium_shape@data,freq_per_postal_code, by = c('POSTCODE' = 'PC'))

#We divide the empirical frequency across the municipalities in Belgium 
#in three bins: low, average and high.
belgium_shape@data$freq_class <- cut(belgium_shape@data$freq, breaks = quantile(belgium_shape@data$freq, 
                                                                                c(0,0.2,0.8,1),na.rm = TRUE),right = FALSE, include.lowest = TRUE, labels = c('low','average','high'))
#Mapping the Belgium with the empirical frequencies 
belgium_shape_f <- fortify(belgium_shape)
belgium_shape_f <- left_join(belgium_shape_f,belgium_shape@data)
plot.eda.map <- ggplot(belgium_shape_f, aes(long,lat, group = group)) +
  geom_polygon(aes(fill = belgium_shape_f$freq_class), colour = 'black', size = 0.1)
plot.eda.map <- plot.eda.map + theme_bw() + labs( fill = 'Empirical\nfrequency') + scale_fill_brewer(palette = 'Blues', na.value = 'White')
plot.eda.map 

####### (iii) Severity 
sev_per_postal_code <- data %>% group_by(PC) %>% summarize(sev = sum(Claim) / sum(NClaims))
belgium_shape = readShapefile()
belgium_shape@data <- left_join(belgium_shape@data,sev_per_postal_code, by = c('POSTCODE' = 'PC'))

#We divide the empirical frequency across the municipalities in Belgium 
#in three bins: low, average and high.
belgium_shape@data$sev_class <- cut(belgium_shape@data$sev, breaks = quantile(belgium_shape@data$sev, 
                                                                                c(0,0.2,0.8,1),na.rm = TRUE),right = FALSE, include.lowest = TRUE, labels = c('low','average','high'))
#Mapping the Belgium with the empirical frequencies 
belgium_shape_f <- fortify(belgium_shape)
belgium_shape_f <- left_join(belgium_shape_f,belgium_shape@data)
plot.eda.map <- ggplot(belgium_shape_f, aes(long,lat, group = group)) +
  geom_polygon(aes(fill = belgium_shape_f$sev_class), colour = 'black', size = 0.1)
plot.eda.map <- plot.eda.map + theme_bw() + labs( fill = 'Empirical\nseverity') + scale_fill_brewer(palette = 'Blues', na.value = 'White')
plot.eda.map 

########### 12. General plots ##############
summary(data_cap$NClaims)
hist_nclaims <- ggplot(data_cap) + geom_bar(aes(x = NClaims), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Number of claims', y = 'Number of policies') + ggtitle('Policies per number of claims')

expo_nclaims <- ggplot(data_cap %>% group_by(NClaims) %>% summarise(totalexpo = sum(Expo)), aes(x = NClaims, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Number of claims', y = 'Total exposure') + ggtitle('The total exposure per number of claims')

grid.arrange(hist_nclaims, expo_nclaims, ncol = 2)

#The number of claims ranges from 0 up to 5. As expected, most policyholders
#have zero claims with exposure and number of policies decreasing in function 
#of the number of claims. 

sum(data_cap$NClaims == 0)/nrow(data_cap)

nclaims_table <- data_cap %>% group_by(NClaims) %>% summarise(totalexposure = sum(Expo), n = n())

hist_expo_bin <- ggplot(data_cap) + geom_bar(aes(x = Expo_bin), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Binned exposure', y = 'Number of policies') + ggtitle('Policies per exposure - binned')
hist_expo_bin

sum(data_cap$Expo == 1)/nrow(data_cap)

#As expected most policies have an exposure of 1. Looking at the 
#histogram for the binned exposures, the number of policies seems to be 
#equal for the other bins. 

summary(data_cap$Claim)
summary(data_cap$Avg_Claim)

#The claim amounts range from 0 up to 1.989.567
#The average claim amounts have the same range. We conclude that 
#this very big claim thus resulted out of 1 claim made by a policyholder.

sum(data_cap$Claim == 0)/nrow(data_cap)
data_claims <- data_cap %>% filter(Claim > 0)
summary(data_claims$Claim)
min(data_claims$Claim)

data_claims <- data_claims %>% arrange(Claim)

data_claims <- data_claims %>% mutate(Id=1:nrow(data_claims))

#Conditioned upon observing a claim. 
#Observing the policies resulting in a claim, it is clear that they also 
#have a very wide range from  1 up to 1.989.567.
#It is clear that this data is right skewed in function of the claims.

p1 <- ggplot(data_claims, aes(x = log(Claim))) + geom_density(colour = "blue") +
  labs(title = "Empirical density of log(claim amounts)", x = "Claim amounts", y = "Empirical density")

p2 <- ggplot(data_claims, aes(x = Claim^(1/3))) + geom_density(colour = "blue") +
  labs(title = "Empirical density of claim amounts^(1/3)", x = "Claim amounts", y = "Empirical density")

grid.arrange(p1, p2, ncol = 2)

p1_avgclaim <- ggplot(data_claims, aes(x = log(Avg_Claim))) + geom_density(colour = "blue") +
  labs(title = "Empirical density of log(average claim amounts)", x = "Average claim amounts", y = "Empirical density")

p2_avgclaim <- ggplot(data_claims, aes(x = Avg_Claim^(1/3))) + geom_density(colour = "blue") +
  labs(title = "Empirical density of average claim amounts^(1/3)", x = "Average claim amounts", y = "Empirical density")

grid.arrange(p1_avgclaim, p2_avgclaim, p1,p2, ncol = 2)

ggplot(data_claims, aes( y =Claim^(1/3))) +        
  geom_boxplot() + labs(y = 'claim amounts^(1/3)', title = 'Boxplot claim amounts')

#Observing the density plots, it is clear that the claim amounts are indeed 
#right skewed. We also observe some peaks in the transformed density plots. 
#We observe two large peaks in the cube root transformation of the claims. 
#Clearly, this is bimodal distributed. 
#It is clear that most claims that are filed are not very big, since 
#the third quantile is 1443.9. Clearly, this will result in a lot of outliers
#for the data. 

################## Correlation study ###############

sel_col <- c("Ageph","BM","Age_car","Power","Sex","Coverage","Fuel","Use","Fleet","region")

#Computing a data set that only contains the covariates used in the models.
#Converting categorical covariates into integers. 
dat_tmp <- data_cap[, sel_col]
dat_tmp <- dat_tmp %>% mutate(Sex = as.integer(Sex), Coverage = as.integer(Coverage),
                              Fuel = as.integer(Fuel), Use = as.integer(Use), Fleet = as.integer(Fleet), 
                              region = as.integer(region))

#Computing the Pearson correlation matrix
corrMat_Pearson <- round(cor(dat_tmp, method = "pearson"), 2)
corrplot(corrMat_Pearson, method = "color")

#Bonus malus and age of driver are negatively correlated, which is intuitive. 
#As the age of the policyholder increases, the bonus malus decreases pointing 
#to a good driver. This observation is indeed logical. Positive correlation 
#between age of the car and the coverage, which again is intuitive. 
#Indeed, we would expect younger cars to have a full coverage whilst 
#older cars have a less extensive insurance such as TPL.

#Computing the Spearman correlation matrix
corrMat_Spearman <- round(cor(dat_tmp, method = "spearman"), 2)

corrplot(corrMat_Spearman ,method = "color")

#Very little correlation between the other variables. 

############# Training and testing data ############
#Computing training and testing data using a 80/20 split.
ind <- partition(y = data_cap[["NClaims"]], p = c(train = 0.8, test = 0.2), seed = seed)
learn <- data_cap[ind$train,]
test <- data_cap[ind$test,]
range(learn$NClaims)
range(test$NClaims)

#Renaming the ID labels
learn <- learn %>% mutate(Id=1:nrow(learn))
test <- test %>% mutate(Id=1:nrow(test))

#Computing the empirical severity of both the training and testing data.
sum(learn$Claim)/sum(learn$NClaims)
sum(test$Claim)/sum(test$NClaims)

sum(learn$Claim)
sum(test$Claim)

#The empirical frequency of the training and testing data. 
sum(learn$NClaims)/sum(learn$Expo)
sum(test$NClaims)/sum(test$Expo)

#Empirical probabilities for the training and testing data. 
freq_learn <- learn %>% group_by(NClaims) %>% summarise(n = n()/nrow(learn))
freq_test <- test %>% group_by(NClaims) %>% summarise(n = n()/nrow(test))

############## Standardization of the data ##################

summary(learn)
str(learn)
summary(test)
str(test)

learn <- learn %>% mutate(
  AgephNN = scale_no_attr(Ageph),
  FuelNN = as.integer(Fuel), 
  FuelNN = scale_no_attr(FuelNN), 
  UseNN = as.integer(Use), 
  UseNN = scale_no_attr(UseNN),
  FleetNN = as.integer(Fleet),
  FleetNN = scale_no_attr(FleetNN), 
  SexNN = as.integer(Sex), 
  SexNN = scale_no_attr(SexNN),
  BMNN = scale_no_attr(BM), 
  Age_carNN = scale_no_attr(Age_car), 
  PowerNN = scale_no_attr(Power)
)



test <- test %>% mutate(
  AgephNN = (Ageph - mean(learn$Ageph, na.rm = TRUE))/sd(learn$Ageph, na.rm = TRUE),
  FuelNN = as.integer(Fuel), 
  FuelNN = (FuelNN - mean(as.integer(learn$Fuel), na.rm = TRUE))/sd(as.integer(learn$Fuel), na.rm = TRUE), 
  UseNN = as.integer(Use), 
  UseNN = (UseNN - mean(as.integer(learn$Use), na.rm = TRUE))/sd(as.integer(learn$Use), na.rm = TRUE),
  FleetNN = as.integer(Fleet),
  FleetNN = (FleetNN - mean(as.integer(learn$Fleet), na.rm = TRUE))/sd(as.integer(learn$Fleet), na.rm = TRUE), 
  SexNN = as.integer(Sex), 
  SexNN = (SexNN - mean(as.integer(learn$Sex), na.rm = TRUE))/sd(as.integer(learn$Sex), na.rm = TRUE),
  BMNN = (BM - mean(learn$BM, na.rm = TRUE))/sd(learn$BM, na.rm = TRUE), 
  Age_carNN = (Age_car - mean(learn$Age_car, na.rm = TRUE))/sd(learn$Age_car, na.rm = TRUE), 
  PowerNN = (Power - mean(learn$Power, na.rm = TRUE))/sd(learn$Power, na.rm = TRUE)
)



#One-hot encoding for the categorical variables with more 
#than 2 levels. In this setting, coverage and region are the only categorical 
#variables with respectively 3 and 9 levels.
#To model coverage we add 3 columns to the dataset. For region we add 9. 

learn <- learn %>% preprocess_cat_onehot("Coverage", "Coverage")
learn <- learn %>% preprocess_cat_onehot("region","region")

test <- test %>% preprocess_cat_onehot("Coverage", "Coverage")
test <- test %>% preprocess_cat_onehot("region","region")





