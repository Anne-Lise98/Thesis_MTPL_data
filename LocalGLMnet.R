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
GDMsteps_localglm <- data.frame(NULL)

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

cp_path <- paste("./Networks/model_localglm1")

cp_callback <- callback_model_checkpoint(
  filepath = cp_path,
  monitor = "val_loss",
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 0
)

#Fitting the model to the learning data. We consider 300 epochs and batch size corresponding to hundred steps in each 
#epoch. Validation data = 0.2 of the learning data.
#Batch size chosen based on the analysis of the out-of-sample losses of the shallow networks
fit_localglm1 <- model_localglm1 %>% fit(
  list(Xlearn, as.matrix(log(learn$Expo))), as.matrix(learn$NClaims),
  epochs = 300,
  batch_size = 1305, validation_split = 0.2, verbose = 1, callbacks = list(cp_callback))

plot(fit_localglm1)

keras_plot_loss_min(fit_localglm1, seed)

load_model_weights_hdf5(model_localglm1, cp_path)



learn$fit_localglm1 <- as.vector(model_localglm1 %>% predict(list(Xlearn, as.matrix(log(learn$Expo)))))
test$fit_localglm1 <- as.vector(model_localglm1 %>% predict(list(Xtest, as.matrix(log(test$Expo)))))

GDMsteps_localglm[1,1] <- 100
GDMsteps_localglm[1,2] <- 1305
GDMsteps_localglm[1,3] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_localglm1)
GDMsteps_localglm[1,4] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_localglm1)
GDMsteps_localglm[1,5] <- Poissondeviance(y_true = learn$NClaims, y_pred = learn$fit_dp11)
GDMsteps_localglm[1,6] <- Poissondeviance(y_true = test$NClaims, y_pred = test$fit_dp11)
GDMsteps_localglm[1,7] <- square_loss(y_true = learn$NClaims, y_pred = learn$fit_dp11)
GDMsteps_localglm[1,8] <- square_loss(y_true = test$NClaims, y_pred = test$fit_dp11)
GDMsteps_localglm[1,9] <- "Two layers - 10 - 20"






