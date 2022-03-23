#Setting the directory
dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dir)
KULbg <- "#116E8A"

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

#Adding new variable indicating the region based on the postcode.
#This variable will simplify the further process.
PC_breaks <- c(1000,2000,3000,4000,5000,6000,7000,8000,9000,10000)

data <- data %>% mutate(region = cut(PC,breaks = PC_breaks, labels = c('1','2','3','4','5','6','7','8','9'),right = FALSE))

#Adding the frequency 
data <- data %>% mutate(freq = NClaims/Expo)

############ First inspection of the data #################

############## 1. Age ################

range(data$Ageph)

#Plotting histogram in function of age policyholders for
#number of policies. There are very little policies 
#for young and old drivers. 
number_policies_Age <- ggplot(data) + geom_bar(aes(x = Ageph), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Age of policyholder', y = 'number of policies') + ggtitle('Policies per age category')

number_policies_Age

sum(data$Ageph < 20)/nrow(data)
sum(data$Ageph >= 90)/nrow(data)

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

freq_age

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
#we will left-censor the age at age 88 -> making sure 
#that this does not distort the model. 

#The empirical severity given that there is a claim 
#for the policyholders per age category. 
#Clearly, since for policyholder with age >92 there are no claims 
#the value for severity will be NaN.
sev_age <- ggplot(data %>% group_by(Ageph) %>% summarise(sev = sum(Claim)/sum(NClaims)), aes(x = Ageph, y = sev)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of policyholder', y = 'Empirical severity') + ggtitle('The empirical severity per age category')

sev_age

#Histogram for the total exposure. Histogram looks like 
#the histogram for the number of policies, which is intuitive.
#As the number of policies for an age group increases
#it is intuitive that the exposure will increase. 
#Furthermore, most policyholders take out policies with 
#exposure of one year.
expo_age <- ggplot(data %>% group_by(Ageph) %>% summarise(totalexpo = sum(Expo)), aes(x = Ageph, y = totalexpo)) + geom_bar(stat = "identity",color = KULbg, fill = "blue", alpha = .5)+ 
  labs(x = 'Age of policyholder', y = 'Total Exposure') + ggtitle('The total exposure per age category')

expo_age


########### General plots ##############
hist_nclaims <- ggplot(data) + geom_bar(aes(x = NClaims), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'number of claims', y = 'number of policies') + ggtitle('Policies per number of claims')

hist_nclaims

hist_expo <- ggplot(data) + geom_bar(aes(x = Expo), color = KULbg, fill = "blue", alpha = .5, stat = "count") + 
  labs(x = 'Exposure', y = 'number of policies') + ggtitle('Policies per exposure')
hist_expo

data %>% group_by(Expo) %>% summarize()