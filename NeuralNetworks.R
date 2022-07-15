#Setting the seed used in the tensorflow package 
seed <- 100
Sys.setenv(PYTHONHASHSEED = seed)
set.seed(seed)
reticulate::py_set_seed(seed)
tensorflow::tf$random$set_seed(seed)

#Homogeneous model 
lambda_hom <- sum(learn$NClaims) / sum(learn$Expo)

#Defining the features used as input in the neural networks. 
#These are the standardized input variables
col_start <- ncol(data_cap) +1 
col_end <- ncol(learn)
features <- c(col_start:col_end)

#Selecting the input for the neural networks, for learning as well as testing data
Xlearn <- as.matrix(learn[, features])  
Xtest <- as.matrix(test[, features])  

#Initializing a data frame containing the evaluated loss functions 
GDMsteps <- data.frame(NULL)

######## Shallow neural networks ########
q0 <- length(features) 
q1 <- 25

###### 1. One GDM step per epoch #####
# 1.1 Fitting a neural network evaluated on a validation set 

#Defining the input layer consisting of the features 
Design1  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol1  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network1 <- Design1 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response1 <- list(Network1, LogVol1) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh <- keras_model(inputs = c(Design1, LogVol1), outputs = c(Response1))
model_sh %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)
summary(model_sh)

#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to one step in each 
#epoch. Validation data = 0.2 of the learning data.
fit1 <- model_sh %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
    epochs = 300,
    batch_size = 130583, validation_split = 0.2, verbose = 1)

plot(fit1)

#Add the fitted data to the learning and testing data sets
learn$fitshNN1 <- as.vector(model_sh %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN1 <- as.vector(model_sh %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#1.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design11  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol11  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network11 <- Design11 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response11 <- list(Network11, LogVol11) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh11 <- keras_model(inputs = c(Design11, LogVol11), outputs = c(Response11))
model_sh11 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 100 epochs, the optimal number of epochs.
#The batch size corresponds to one step in each epoch. 
fit11 <- model_sh11 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 130583, verbose = 1)

plot(fit11)

#Add the fitted data to the learning and testing data sets
learn$fitshNN11 <- as.vector(model_sh11 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN11 <- as.vector(model_sh11 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#Adding the loss values to the data frame
GDMsteps[1,1] <- 1 
GDMsteps[1,2] <- 130583
GDMsteps[1,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN1)
GDMsteps[1,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN1)
GDMsteps[1,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN11)
GDMsteps[1,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN11)
GDMsteps[1,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fitshNN11)
GDMsteps[1,8] <- square_loss(y_true = test$NClaims, y_pred = test$fitshNN11)

colnames(GDMsteps) <- c("GDM steps", "Batch size", "Initial in-sample Poisson loss", "Initial out-of-sample Poisson loss", "In-sample Poisson loss", "Out-of-sample Poisson loss", "In-sample square loss", "Out-of-sample square loss")


###### 2. Five GDM steps per epoch #####
# 2.1 Fitting a neural network evaluated on a validation set 

#Defining the input layer consisting of the features 
Design2  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol2  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network2 <- Design2 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response2 <- list(Network2, LogVol2) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh2 <- keras_model(inputs = c(Design2, LogVol2), outputs = c(Response2))
model_sh2 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to five steps in each 
#epoch. Validation data = 0.2 of the learning data.
fit2 <- model_sh2 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 26116, validation_split = 0.2, verbose = 1)

plot(fit2)

#Add the fitted data to the learning and testing data sets
learn$fitshNN2 <- as.vector(model_sh2 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN2 <- as.vector(model_sh2 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#2.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design21  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol21  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network21 <- Design21 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response21 <- list(Network21, LogVol21) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh21 <- keras_model(inputs = c(Design21, LogVol21), outputs = c(Response21))
model_sh21 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 120 epochs, the optimal number of epochs.
#The batch size corresponds to five steps in each epoch. 
fit21 <- model_sh21 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 120,
  batch_size = 26116, verbose = 1)

plot(fit21)

#Add the fitted data to the learning and testing data sets
learn$fitshNN21 <- as.vector(model_sh21 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN21 <- as.vector(model_sh21 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#Adding the loss values to the data frame
GDMsteps[2,1] <- 5 
GDMsteps[2,2] <- 26116
GDMsteps[2,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN2)
GDMsteps[2,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN2)
GDMsteps[2,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN21)
GDMsteps[2,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN21)
GDMsteps[2,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fitshNN21)
GDMsteps[2,8] <- square_loss(y_true = test$NClaims, y_pred = test$fitshNN21)

###### 3. Ten GDM steps per epoch #####
# 3.1 Fitting a neural network evaluated on a validation set 

#Defining the input layer consisting of the features 
Design3  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol3 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network3 <- Design3 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response3 <- list(Network3, LogVol3) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh3 <- keras_model(inputs = c(Design3, LogVol3), outputs = c(Response3))
model_sh3 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to ten steps in each 
#epoch. Validation data = 0.2 of the learning data.
fit3 <- model_sh3 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 13058, validation_split = 0.2, verbose = 1)

plot(fit3)

#Add the fitted data to the learning and testing data sets
learn$fitshNN3 <- as.vector(model_sh3 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN3 <- as.vector(model_sh3 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#3.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design31  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol31  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network31 <- Design31 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response31 <- list(Network31, LogVol31) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh31 <- keras_model(inputs = c(Design31, LogVol31), outputs = c(Response31))
model_sh31 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 75 epochs, the optimal number of epochs.
#The batch size corresponds to ten steps in each epoch. 
fit31 <- model_sh31 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 75,
  batch_size = 13058, verbose = 1)

plot(fit31)

#Add the fitted data to the learning and testing data sets
learn$fitshNN31 <- as.vector(model_sh31 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN31 <- as.vector(model_sh31 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#Adding the loss values to the data frame
GDMsteps[3,1] <- 10
GDMsteps[3,2] <- 13058
GDMsteps[3,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN3)
GDMsteps[3,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN3)
GDMsteps[3,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN31)
GDMsteps[3,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN31)
GDMsteps[3,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fitshNN31)
GDMsteps[3,8] <- square_loss(y_true = test$NClaims, y_pred = test$fitshNN31)


###### 4. Fifty GDM steps per epoch #####
# 4.1 Fitting a neural network evaluated on a validation set 

#Defining the input layer consisting of the features 
Design4  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol4 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network4 <- Design4 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response4 <- list(Network4, LogVol4) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh4 <- keras_model(inputs = c(Design4, LogVol4), outputs = c(Response4))
model_sh4 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to fifty steps in each 
#epoch. Validation data = 0.2 of the learning data.
fit4 <- model_sh4 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 2611, validation_split = 0.2, verbose = 1)

plot(fit4)

#Add the fitted data to the learning and testing data sets
learn$fitshNN4 <- as.vector(model_sh4 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN4 <- as.vector(model_sh4 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#4.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design41  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol41  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network41 <- Design41 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response41 <- list(Network41, LogVol41) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh41 <- keras_model(inputs = c(Design41, LogVol41), outputs = c(Response41))
model_sh41 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 50 epochs, the optimal number of epochs.
#The batch size corresponds to fifty steps in each epoch. 
fit41 <- model_sh41 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 50,
  batch_size = 2611, verbose = 1)

plot(fit41)

#Add the fitted data to the learning and testing data sets
learn$fitshNN41 <- as.vector(model_sh41 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN41 <- as.vector(model_sh41 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#Adding the loss values to the data frame
GDMsteps[4,1] <- 50
GDMsteps[4,2] <- 2611
GDMsteps[4,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN4)
GDMsteps[4,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN4)
GDMsteps[4,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN41)
GDMsteps[4,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN41)
GDMsteps[4,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fitshNN41)
GDMsteps[4,8] <- square_loss(y_true = test$NClaims, y_pred = test$fitshNN41)

###### 5. Hundred GDM steps per epoch #####
# 5.1 Fitting a neural network evaluated on a validation set 

#Defining the input layer consisting of the features 
Design5  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol5 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network5 <- Design5 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response5 <- list(Network5, LogVol5) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh5 <- keras_model(inputs = c(Design5, LogVol5), outputs = c(Response5))
model_sh5 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to hundred steps in each 
#epoch. Validation data = 0.2 of the learning data.
fit5 <- model_sh5 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 1305, validation_split = 0.2, verbose = 1)

plot(fit5)

#Add the fitted data to the learning and testing data sets
learn$fitshNN5 <- as.vector(model_sh5 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN5 <- as.vector(model_sh5 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#5.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design51  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol51  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network51 <- Design51 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response51 <- list(Network51, LogVol51) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh51 <- keras_model(inputs = c(Design51, LogVol51), outputs = c(Response51))
model_sh51 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 50 epochs, the optimal number of epochs.
#The batch size corresponds to hundred steps in each epoch. 
fit51 <- model_sh51 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 50,
  batch_size = 1305, verbose = 1)

plot(fit51)

#Add the fitted data to the learning and testing data sets
learn$fitshNN51 <- as.vector(model_sh51 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN51 <- as.vector(model_sh51 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#Adding the loss values to the data frame
GDMsteps[5,1] <- 100
GDMsteps[5,2] <- 1305
GDMsteps[5,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN5)
GDMsteps[5,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN5)
GDMsteps[5,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN51)
GDMsteps[5,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN51)
GDMsteps[5,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fitshNN51)
GDMsteps[5,8] <- square_loss(y_true = test$NClaims, y_pred = test$fitshNN51)


###### 6. Five hundred GDM steps per epoch #####
# 6.1 Fitting a neural network evaluated on a validation set 

#Defining the input layer consisting of the features 
Design6  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol6 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network6 <- Design6 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response6 <- list(Network6, LogVol6) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh6 <- keras_model(inputs = c(Design6, LogVol6), outputs = c(Response6))
model_sh6 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to five hundred steps in each 
#epoch. Validation data = 0.2 of the learning data.
fit6 <- model_sh6 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 261, validation_split = 0.2, verbose = 1)

plot(fit6)

#Add the fitted data to the learning and testing data sets
learn$fitshNN6 <- as.vector(model_sh6 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN6 <- as.vector(model_sh6 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#6.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design61  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol61  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network61 <- Design61 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network',
              weights = list(array(0, dim = c(q1, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response61 <- list(Network61, LogVol61) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_sh61 <- keras_model(inputs = c(Design61, LogVol61), outputs = c(Response61))
model_sh61 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider ten epochs, the optimal number of epochs.
#The batch size corresponds to five hundred steps in each epoch. 
fit61 <- model_sh61 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 10,
  batch_size = 261, verbose = 1)

plot(fit61)

#Add the fitted data to the learning and testing data sets
learn$fitshNN61 <- as.vector(model_sh61 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fitshNN61 <- as.vector(model_sh61 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#Adding the loss values to the data frame
GDMsteps[6,1] <- 500
GDMsteps[6,2] <- 261
GDMsteps[6,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN6)
GDMsteps[6,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN6)
GDMsteps[6,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fitshNN61)
GDMsteps[6,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fitshNN61)
GDMsteps[6,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fitshNN61)
GDMsteps[6,8] <- square_loss(y_true = test$NClaims, y_pred = test$fitshNN61)


########## Deep neural network #########

####### 1. Neural network with 2 hidden layers ###### 
q0 <- length(features)   # dimension of features
q1 <- 10                # number of neurons in first hidden layer
q2 <- 20                 # number of neurons in second hidden layer
        
#Defining the input layer consisting of the features 
Design_dp1 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_dp1 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network, focusing on the features 
Network_dp1 <- Design_dp1 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network', 
              weights = list(array(0, dim = c(q2, 1)), array(log(lambda_hom), dim = c(1))))


#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response_dp1 <- list(Network_dp1, LogVol_dp1) %>%
  layer_add(name = 'Add') %>% 
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_dp1 <- keras_model(inputs = c(Design_dp1, LogVol_dp1), outputs = c(Response_dp1))
model_dp1 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to hundred steps in each 
#epoch. Validation data = 0.2 of the learning data.
#Batch size chosen based on the analysis of the out-of-sample losses of the shallow networks
fit_dp1 <- model_dp1 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 1305, validation_split = 0.2, verbose = 1)

plot(fit_dp1)

#Add the fitted data to the learning and testing data sets
learn$fit_dp1 <- as.vector(model_dp1 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_dp1 <- as.vector(model_dp1 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#1.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design_dp11  <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_dp11  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network_dp11 <- Design_dp11 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network', 
              weights = list(array(0, dim = c(q2, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response_dp11 <- list(Network_dp11, LogVol_dp11) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_dp11 <- keras_model(inputs = c(Design_dp11, LogVol_dp11), outputs = c(Response_dp11))
model_dp11 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 25 epochs, the optimal number of epochs.
#The batch size corresponds to hundred steps in each epoch. 
fit_dp11 <- model_dp11 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 25,
  batch_size = 1305, verbose = 1)
plot(fit_dp11)

#Add the fitted data to the learning and testing data sets
learn$fit_dp11 <- as.vector(model_dp11 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_dp11 <- as.vector(model_dp11 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#Adding the loss values to the data frame
for(i in 1:6){
  GDMsteps[i,9] <- "Shallow"
}

colnames(GDMsteps)[9] <- "Neural network specification" 

GDMsteps[7,1] <- 100
GDMsteps[7,2] <- 1305
GDMsteps[7,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_dp1)
GDMsteps[7,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_dp1)
GDMsteps[7,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_dp11)
GDMsteps[7,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_dp11)
GDMsteps[7,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_dp11)
GDMsteps[7,8] <- square_loss(y_true = test$NClaims, y_pred = test$fit_dp11)
GDMsteps[7,9] <- "Two layers - 10 - 20"

####### 2. Neural network with 3 hidden layers ###### 
q0 <- length(features)   # dimension of features
q1 <- 20               # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in third hidden layer

#Defining the input layer consisting of the features 
Design_dp2 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_dp2 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network, focusing on the features 
Network_dp2 <- Design_dp2 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network', 
              weights = list(array(0, dim = c(q3, 1)), array(log(lambda_hom), dim = c(1))))


#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response_dp2 <- list(Network_dp2, LogVol_dp2) %>%
  layer_add(name = 'Add') %>% 
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_dp2 <- keras_model(inputs = c(Design_dp2, LogVol_dp2), outputs = c(Response_dp2))
model_dp2 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)
summary(model_dp2)
#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to hundred steps in each 
#epoch. Validation data = 0.2 of the learning data.
#Batch size chosen based on the analysis of the out-of-sample losses of the shallow networks
fit_dp2 <- model_dp2 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 1305, validation_split = 0.2, verbose = 1)

plot(fit_dp2)

#Add the fitted data to the learning and testing data sets
learn$fit_dp2 <- as.vector(model_dp2 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_dp2 <- as.vector(model_dp2 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#2.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design_dp21 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_dp21  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network_dp21 <- Design_dp21 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network', 
              weights = list(array(0, dim = c(q3, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response_dp21 <- list(Network_dp21, LogVol_dp21) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_dp21 <- keras_model(inputs = c(Design_dp21, LogVol_dp21), outputs = c(Response_dp21))
model_dp21 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 25 epochs, the optimal number of epochs.
#The batch size corresponds to hundred steps in each epoch. 
fit_dp21 <- model_dp21 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 25,
  batch_size = 1305, verbose = 1)
plot(fit_dp21)

#Add the fitted data to the learning and testing data sets
learn$fit_dp21 <- as.vector(model_dp21 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_dp21 <- as.vector(model_dp21 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))


GDMsteps[8,1] <- 100
GDMsteps[8,2] <- 1305
GDMsteps[8,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_dp2)
GDMsteps[8,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_dp2)
GDMsteps[8,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_dp21)
GDMsteps[8,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_dp21)
GDMsteps[8,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_dp21)
GDMsteps[8,8] <- square_loss(y_true = test$NClaims, y_pred = test$fit_dp21)
GDMsteps[8,9] <- "Three layers - 20 - 15 - 10"

####### 3. Neural network with 3 hidden layers ###### 
q0 <- length(features)   # dimension of features
q1 <- 10               # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in third hidden layer

#Defining the input layer consisting of the features 
Design_dp3 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_dp3 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network, focusing on the features 
Network_dp3 <- Design_dp3 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network', 
              weights = list(array(0, dim = c(q3, 1)), array(log(lambda_hom), dim = c(1))))


#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response_dp3 <- list(Network_dp3, LogVol_dp3) %>%
  layer_add(name = 'Add') %>% 
  layer_dense(units = 1, activation = k_exp, name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_dp3 <- keras_model(inputs = c(Design_dp3, LogVol_dp3), outputs = c(Response_dp3))
model_dp3 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)
summary(model_dp3)
#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to hundred steps in each 
#epoch. Validation data = 0.2 of the learning data.
#Batch size chosen based on the analysis of the out-of-sample losses of the shallow networks
fit_dp3 <- model_dp3 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 1305, validation_split = 0.2, verbose = 1)

plot(fit_dp3)

#Add the fitted data to the learning and testing data sets
learn$fit_dp3 <- as.vector(model_dp3 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_dp3 <- as.vector(model_dp3 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

#3.2 Fixing optimal number of epochs based on the previous fit

#Defining the input layer consisting of the features 
Design_dp31 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_dp31  <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')

#Defining the neural network, focusing on the features 
Network_dp31 <- Design_dp31 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network', 
              weights = list(array(0, dim = c(q3, 1)), array(log(lambda_hom), dim = c(1))))

#Defining the total neural network, putting the features and offset together 
#We do not need to train the weights since the weights corresponding to the features 
#have been trained in the previous step. 
#Exponential activation function since we consider the Poisson regression model
Response_dp31 <- list(Network_dp31, LogVol_dp31) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_dp31 <- keras_model(inputs = c(Design_dp31, LogVol_dp31), outputs = c(Response_dp31))
model_dp31 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

#Fitting the model to the learning data. We consider 30 epochs, the optimal number of epochs.
#The batch size corresponds to hundred steps in each epoch. 
fit_dp31 <- model_dp31 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 30,
  batch_size = 1305, verbose = 1)
plot(fit_dp31)
#Add the fitted data to the learning and testing data sets
learn$fit_dp31 <- as.vector(model_dp31 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_dp31 <- as.vector(model_dp31 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))


GDMsteps[9,1] <- 100
GDMsteps[9,2] <- 1305
GDMsteps[9,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_dp3)
GDMsteps[9,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_dp3)
GDMsteps[9,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_dp31)
GDMsteps[9,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_dp31)
GDMsteps[9,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_dp31)
GDMsteps[9,8] <- square_loss(y_true = test$NClaims, y_pred = test$fit_dp31)
GDMsteps[9,9] <- "Three layers - 10 - 15 - 10"





