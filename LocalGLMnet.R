#Setting the seed used in the tensorflow package 
seed <- 100
Sys.setenv(PYTHONHASHSEED = seed)
set.seed(seed)
reticulate::py_set_seed(seed)
tensorflow::tf$random$set_seed(seed)

#Homogeneous model 
lambda_hom <- sum(learn$NClaims) / sum(learn$Expo)

#Defining the features used as input in the localglmnet. 
#These are the standardized input variables.
col_start <- ncol(data_cap) +1 
col_end <- ncol(learn)
features <- c(col_start:col_end)

#Selecting the input for the localglmnet, for learning as well as testing data.
Xlearn <- as.matrix(learn[, features])  
Xtest <- as.matrix(test[, features])  

#Initializing a data frame containing the evaluated loss functions. 
GDMsteps_localglm <- data.frame(NULL)

#### 1. Localglmnet: regression attentions with two hidden layers #####
###### 1.1 Batch size = 3000 ####
q0 <- length(features)   # dimension of features
q1 <- 10                # number of neurons in first hidden layer
q2 <- 20                #number of neurons in second hidden layer 


#Defining the input layer consisting of the features 
Design_localglm1 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm1 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm1 <- Design_localglm1 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm1 <- list(Design_localglm1, Attention_localglm1) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm1 <- list(Linear_localglm1, LogVol_localglm1) %>% layer_add(name = 'Add') %>% layer_dense(
    units=1, activation='exponential', name='output', trainable = FALSE,
    weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm1 <- keras_model(inputs = c(Design_localglm1, LogVol_localglm1), outputs = c(Response_localglm1))
model_localglm1 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm1)

#Define a path
cp_path <- paste("./Networks/model_localglm1")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 3000.
# Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm1 <- model_localglm1 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 3000, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm1)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm1, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm1, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm1 <- as.vector(model_localglm1 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm1 <- as.vector(model_localglm1 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[1,1] <- 44
GDMsteps_localglm[1,2] <- 3000
GDMsteps_localglm[1,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm1)
GDMsteps_localglm[1,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm1)
GDMsteps_localglm[1,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm1)
GDMsteps_localglm[1,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm1)
GDMsteps_localglm[1,7] <- "Two layers - 10 - 20"

colnames(GDMsteps_localglm) <- c("GDM steps per epoch", "Batch size", "In-sample Poisson loss", "Out-of-sample Poisson loss",
                                 "In-sample square loss", "Out-of-sample square loss", "Network specification")

###### 1.2 Batch size = 13058 ####
q0 <- length(features)   # dimension of features
q1 <- 10                # number of neurons in first hidden layer
q2 <- 20                #number of neurons in second hidden layer 


#Defining the input layer consisting of the features 
Design_localglm11 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm11 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm11 <- Design_localglm11 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm11 <- list(Design_localglm11, Attention_localglm11) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                      activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm11 <- list(Linear_localglm11, LogVol_localglm11) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm11 <- keras_model(inputs = c(Design_localglm11, LogVol_localglm11), outputs = c(Response_localglm11))
model_localglm11 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm11)

#Define a path
cp_path <- paste("./Networks/model_localglm11")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 200 epochs and batch size = 13058.
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm11 <- model_localglm11 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 200,
  batch_size = 13058, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm11)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm11, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm11, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm11 <- as.vector(model_localglm11 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm11 <- as.vector(model_localglm11 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[2,1] <- 10
GDMsteps_localglm[2,2] <- 13058
GDMsteps_localglm[2,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm11)
GDMsteps_localglm[2,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm11)
GDMsteps_localglm[2,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm11)
GDMsteps_localglm[2,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm11)
GDMsteps_localglm[2,7] <- "Two layers - 10 - 20"

###### 1.3 Batch size = 1305 ####
q0 <- length(features)   # dimension of features
q1 <- 10                # number of neurons in first hidden layer
q2 <- 20                #number of neurons in second hidden layer 


#Defining the input layer consisting of the features 
Design_localglm12 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm12 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm12 <- Design_localglm12 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm12 <- list(Design_localglm12, Attention_localglm12) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                         activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm12 <- list(Linear_localglm12, LogVol_localglm12) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm12 <- keras_model(inputs = c(Design_localglm12, LogVol_localglm12), outputs = c(Response_localglm12))
model_localglm12 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm12)

#Define a path
cp_path <- paste("./Networks/model_localglm12")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 1305
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm12 <- model_localglm12 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 1305, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm12)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm12, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm12, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm12 <- as.vector(model_localglm12 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm12 <- as.vector(model_localglm12 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[3,1] <- 100
GDMsteps_localglm[3,2] <- 1305
GDMsteps_localglm[3,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm12)
GDMsteps_localglm[3,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm12)
GDMsteps_localglm[3,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm12)
GDMsteps_localglm[3,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm12)
GDMsteps_localglm[3,7] <- "Two layers - 10 - 20"

#### 2. Localglmnet: regression attentions with three hidden layers #####
##### 2.1 Batch size = 3000 ####
q0 <- length(features)   # dimension of features
q1 <- 20               # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in third hidden layer


#Defining the input layer consisting of the features 
Design_localglm2 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm2 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm2 <- Design_localglm2 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm2 <- list(Design_localglm2, Attention_localglm2) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                      activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm2 <- list(Linear_localglm2, LogVol_localglm2) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm2 <- keras_model(inputs = c(Design_localglm2, LogVol_localglm2), outputs = c(Response_localglm2))
model_localglm2 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm2)

#Define a path
cp_path <- paste("./Networks/model_localglm2")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 3000. 
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm2 <- model_localglm2 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 3000, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm2)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm2, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm2, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm2 <- as.vector(model_localglm2 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm2 <- as.vector(model_localglm2 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[4,1] <- 100
GDMsteps_localglm[4,2] <- 3000
GDMsteps_localglm[4,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm2)
GDMsteps_localglm[4,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm2)
GDMsteps_localglm[4,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm2)
GDMsteps_localglm[4,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm2)
GDMsteps_localglm[4,7] <- "Three layers - 20 - 15 - 10"

#### 3.2 Batch size = 1305 ####
q0 <- length(features)   # dimension of features
q1 <- 20               # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in third hidden layer


#Defining the input layer consisting of the features 
Design_localglm21 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm21 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm21 <- Design_localglm21 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm21 <- list(Design_localglm21, Attention_localglm21) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                      activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm21 <- list(Linear_localglm21, LogVol_localglm21) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm21 <- keras_model(inputs = c(Design_localglm21, LogVol_localglm21), outputs = c(Response_localglm21))
model_localglm21 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm2)

#Define a path
cp_path <- paste("./Networks/model_localglm21")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 1305. 
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm21 <- model_localglm21 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 1305, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm21)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm21, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm21, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm21 <- as.vector(model_localglm21 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm21 <- as.vector(model_localglm21 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[5,1] <- 100
GDMsteps_localglm[5,2] <- 1305
GDMsteps_localglm[5,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm21)
GDMsteps_localglm[5,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm21)
GDMsteps_localglm[5,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm21)
GDMsteps_localglm[5,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm21)
GDMsteps_localglm[5,7] <- "Three layers - 20 - 15 - 10"

#### 3. Localglmnet: regression attentions with three hidden layers #####
#### 3.1 Batch size = 3000 ####
q0 <- length(features)   # dimension of features
q1 <- 10               # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in third hidden layer


#Defining the input layer consisting of the features 
Design_localglm3 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm3 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm3 <- Design_localglm3 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm3 <- list(Design_localglm3, Attention_localglm3) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm3 <- list(Linear_localglm3, LogVol_localglm3) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm3 <- keras_model(inputs = c(Design_localglm3, LogVol_localglm3), outputs = c(Response_localglm3))
model_localglm3 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm3)

#Define a path
cp_path <- paste("./Networks/model_localglm3")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 3000.
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm3 <- model_localglm3 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 3000, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm3)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm3, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm3, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm3 <- as.vector(model_localglm3 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm3 <- as.vector(model_localglm3 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[6,1] <- 100
GDMsteps_localglm[6,2] <- 3000
GDMsteps_localglm[6,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm3)
GDMsteps_localglm[6,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm3)
GDMsteps_localglm[6,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm3)
GDMsteps_localglm[6,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm3)
GDMsteps_localglm[6,7] <- "Three layers - 10 - 15 - 10"

#### 3.2 Batch size = 1305 ####
q0 <- length(features)   # dimension of features
q1 <- 10               # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in third hidden layer


#Defining the input layer consisting of the features 
Design_localglm31 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm31 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm31 <- Design_localglm31 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm31 <- list(Design_localglm31, Attention_localglm31) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                      activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm31 <- list(Linear_localglm31, LogVol_localglm31) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm31 <- keras_model(inputs = c(Design_localglm31, LogVol_localglm31), outputs = c(Response_localglm31))
model_localglm31 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm31)

#Define a path
cp_path <- paste("./Networks/model_localglm31")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 1305.
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm31 <- model_localglm31 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 1305, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm31)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm31, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm31, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm31 <- as.vector(model_localglm31 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm31 <- as.vector(model_localglm31 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[7,1] <- 100
GDMsteps_localglm[7,2] <- 1305
GDMsteps_localglm[7,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm31)
GDMsteps_localglm[7,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm31)
GDMsteps_localglm[7,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm31)
GDMsteps_localglm[7,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm31)
GDMsteps_localglm[7,7] <- "Three layers - 10 - 15 - 10"

#### 4. Localglmnet: regression attentions with one hidden layer #####
#### 4.1 Batch size = 3000 ####
q0 <- length(features)   # dimension of features
q1 <- 25             # number of neurons in first hidden layer
               


#Defining the input layer consisting of the features 
Design_localglm4 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm4 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm4 <- Design_localglm4 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm4 <- list(Design_localglm4, Attention_localglm4) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                      activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm4 <- list(Linear_localglm4, LogVol_localglm4) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm4 <- keras_model(inputs = c(Design_localglm4, LogVol_localglm4), outputs = c(Response_localglm4))
model_localglm4 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm4)

#Define a path
cp_path <- paste("./Networks/model_localglm4")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 3000. 
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm4 <- model_localglm4 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 3000, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm4)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm4, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm4, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm4 <- as.vector(model_localglm4 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm4 <- as.vector(model_localglm4 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[8,1] <- 100
GDMsteps_localglm[8,2] <- 3000
GDMsteps_localglm[8,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm4)
GDMsteps_localglm[8,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm4)
GDMsteps_localglm[8,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm4)
GDMsteps_localglm[8,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm4)
GDMsteps_localglm[8,7] <- "One layer - 25"

#### 4.1 Batch size = 1305 ####
q0 <- length(features)   # dimension of features
q1 <- 25             # number of neurons in first hidden layer



#Defining the input layer consisting of the features 
Design_localglm41 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm41 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm41 <- Design_localglm41 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm41 <- list(Design_localglm41, Attention_localglm41) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                      activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm41 <- list(Linear_localglm41, LogVol_localglm41) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm41 <- keras_model(inputs = c(Design_localglm41, LogVol_localglm41), outputs = c(Response_localglm41))
model_localglm41 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm41)

#Define a path
cp_path <- paste("./Networks/model_localglm41")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 1305. 
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm41 <- model_localglm41 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 1305, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm41)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm41, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm41, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm41 <- as.vector(model_localglm41 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm41 <- as.vector(model_localglm41 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[9,1] <- 100
GDMsteps_localglm[9,2] <- 1305
GDMsteps_localglm[9,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm41)
GDMsteps_localglm[9,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm41)
GDMsteps_localglm[9,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm41)
GDMsteps_localglm[9,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm41)
GDMsteps_localglm[9,7] <- "One layer - 25"

#### 5. Localglmnet: regression attentions with three hidden layers #####
#### Batch size = 3000 ####
q0 <- length(features)   # dimension of features
q1 <- 25               # number of neurons in first hidden layer
q2 <- 20                 # number of neurons in second hidden layer
q3 <- 15                # number of neurons in third hidden layer


#Defining the input layer consisting of the features 
Design_localglm5 <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm5 <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm5 <- Design_localglm5 %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm5 <- list(Design_localglm5, Attention_localglm5) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                      activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm5 <- list(Linear_localglm5, LogVol_localglm5) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm5 <- keras_model(inputs = c(Design_localglm5, LogVol_localglm5), outputs = c(Response_localglm5))
model_localglm5 %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)


#Define a path
cp_path <- paste("./Networks/model_localglm5")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 3000.
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm5 <- model_localglm5 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 3000, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm5)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm5, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm5, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm5 <- as.vector(model_localglm5 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm5 <- as.vector(model_localglm5 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[10,1] <- 100
GDMsteps_localglm[10,2] <- 3000
GDMsteps_localglm[10,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm5)
GDMsteps_localglm[10,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm5)
GDMsteps_localglm[10,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm5)
GDMsteps_localglm[10,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm5)
GDMsteps_localglm[10,7] <- "Three layers - 25 - 20 - 15"


###### 6. Variable importance and selection ####

#Defining a vector consisting of the features used in the localglmnet. 
#In this case, we include the two random features RandNN and RandUN.
col_features <- c("AgephNN", "FuelNN", "UseNN", "FleetNN", "SexNN", "BMNN", "Age_carNN", "PowerNN","Coverage1", "Coverage2", "Coverage3", "region1",
                  "region2", "region3", "region4", "region5","region6", "region7", "region8", "region9", "RandNN", "RandUN") 

col_names <- c("Ageph", "Fuel", "Use", "Fleet", "Sex", "BM", "Age_car", "Power","Coverage1", "Coverage2", "Coverage3", "region1",
                  "region2", "region3", "region4", "region5","region6", "region7", "region8", "region9", "RandNN", "RandUN") 

#Defining the number of neurons for the hidden layers. 
q0 <- length(col_features)   # dimension of features
q1 <- 10               # number of neurons in first hidden layer
q2 <- 15                 # number of neurons in second hidden layer
q3 <- 10                 # number of neurons in third hidden layer


XX <- as.matrix(learn[, col_features])
TT <- as.matrix(test[, col_features])

#Defining the input layer consisting of the features 
Design_localglm_selection <- layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 

#Defining the input layer consisting of the offset 
LogVol_localglm_selection <- layer_input(shape = c(1), dtype = 'float32', name = 'LogVol')


#Defining the neural network for the attentions. 
#Clearly, the input dimension = output dimension
Attention_localglm_selection <- Design_localglm_selection %>%
  layer_dense(units = q1, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = q2, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = q3, activation = 'tanh', name = 'layer3') %>%
  layer_dense(units = q0, activation = 'linear', name = 'attention')

#Defining the layer containing the elementwise product of the attentions and features. 
#Defining the initial value for the bias by the homogeneous model.
Linear_localglm_selection <- list(Design_localglm_selection, Attention_localglm_selection) %>% layer_dot(name='product', axes=1) %>% layer_dense(units = 1,
                                                                                                                         activation = 'linear', name = 'Input', weights = list(array(0, dim = c(1, 1)), array(log(lambda_hom), dim = c(1))))

#We put the linear layer and the offset together. As the weights have been trained in the previous step, 
#we do not train the weights here and fix bias == 0. 
#Exponential activation function since we consider the Poisson regression model. 
Response_localglm_selection <- list(Linear_localglm_selection, LogVol_localglm_selection) %>% layer_add(name = 'Add') %>% layer_dense(
  units=1, activation='exponential', name='output', trainable = FALSE,
  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))

#Defining the model
model_localglm_selection <- keras_model(inputs = c(Design_localglm_selection, LogVol_localglm_selection), outputs = c(Response_localglm_selection))
model_localglm_selection %>% compile(
  loss = 'poisson',
  optimizer = "nadam"
)

summary(model_localglm_selection)

#Define a path
cp_path <- paste("./Networks/model_localglm_selection")

#Define a callback to obtain the weights of the model that minimizes the validation loss. 
#We only save the weights corresponding to the best model.
cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 100 epochs and batch size = 1305.
#Validation data = 0.2 of the learning data.
#Callback included to obtain the weights corresponding to the best model (model that minimizes validation loss).
fit_localglm_selection <- model_localglm_selection %>% fit(
  list(XX, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 100,
  batch_size = 1305, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm_selection)

#Plot validation loss and training loss, including line that indicates the minimum validation loss. 
keras_plot_loss_min(fit_localglm_selection, seed)

#Load the weights for the model corresponding to the smallest validation loss. 
load_model_weights_hdf5(model_localglm_selection, cp_path)

#Add the fitted data to the learning and testing data sets
learn$fit_localglm_selection <- as.vector(model_localglm_selection %>% predict(list(XX, as.matrix(log(learn$Expo)))))
test$fit_localglm_selection <- as.vector(model_localglm_selection %>% predict(list(TT, as.matrix(log(test$Expo)))))

GDMsteps_localglm[11,1] <- 100
GDMsteps_localglm[11,2] <- 1305
GDMsteps_localglm[11,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm_selection)
GDMsteps_localglm[11,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm_selection)
GDMsteps_localglm[11,5] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_localglm_selection)
GDMsteps_localglm[11,6] <- square_loss(y_true = test$NClaims, y_pred = test$fit_localglm_selection)
GDMsteps_localglm[11,7] <- "Three layers - 10 - 15 - 10"

#Selecting the submodel that only exists of the unweighted regression attentions
zz <- keras_model(inputs = model_localglm_selection$input[[1]], outputs = get_layer(model_localglm_selection, 'attention')$output)
summary(zz)

#Computing the unweighted regression attentions on the testing data.
beta_x <- data.frame(zz %>% predict(list(TT)))

#Selecting the weight corresponding to the linear layer.
ww <- as.numeric(get_weights(model_localglm_selection)[[9]])

#Computing the real regression attentions for the testing data
beta <- beta_x*ww 
names(beta) <- paste("Beta",col_names)

#Computing the regression attentions for the learning data 
beta_xL <- data.frame(zz %>% predict(list(XX)))
beta_L <- beta_xL*ww 
names(beta_L) <- paste("Beta",col_names)

