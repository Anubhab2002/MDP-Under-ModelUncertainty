import json
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
import keras
from keras import layers
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import random
from tqdm import tqdm
from constants import ticker_list, training_end, test_periods

test_period_1_start, test_period_1_end = test_periods[0]
test_period_2_start, test_period_2_end = test_periods[1]
test_period_3_start, test_period_3_end = test_periods[2]

# Load data from CSVs into a list of DataFrames
stocks = [pd.read_csv(f"/teamspace/studios/this_studio/FinMath_Project/{ticker}.csv") for ticker in ticker_list]

# Number of periods to look back for returns
periods = 10  # Lookback period (m)

# Calculate 10-period rolling returns for each stock
Returns_10periods = []

for stock in stocks:
    # Ensure the 'Close' column is numeric
    stock["Close"] = pd.to_numeric(stock["Close"], errors="coerce")
    stock.dropna(subset=["Close"], inplace=True)  # Remove rows with NaN in 'Close'

    # Calculate 10-period returns
    stock_returns = [
        [
            (stock["Close"].iloc[j + k] - stock["Close"].iloc[j - 1 + k]) / stock["Close"].iloc[j - 1 + k]
            for k in range(periods)
            if j + k < len(stock)  # Ensure j + k does not exceed bounds
        ]
        for j in range(1, len(stock) - periods + 1)
    ]
    
    Returns_10periods.append(stock_returns)


# Convert Returns_10periods to a TensorFlow constant for efficient handling in models
Returns_10periods = tf.constant(Returns_10periods, dtype=tf.float32)

# Split into training and testing sets using tf.gather
Returns_10periods_train = tf.gather(Returns_10periods, range(training_end), axis=1)
Returns_10periods_test_1 = tf.gather(Returns_10periods, range(test_period_1_start, test_period_1_end), axis=1)
Returns_10periods_test_2 = tf.gather(Returns_10periods, range(test_period_2_start, test_period_2_end), axis=1)
Returns_10periods_test_3 = tf.gather(Returns_10periods, range(test_period_3_start, test_period_3_end), axis=1)

class Robust_Portfolio_Optimization:
    def __init__(self,
                 Returns_train, # Returns used for the training (Consist of at least 2 period-returns)
                 uncertainty = "Wasserstein", # Other option: Parametric
                 C = 1, # Bound for the investment
                 sigma = 0.01, # Volatility of the asset returns for the batches
                   epsilon = 0.01, # Epsilon in the definition of the ambiguity sets                   
                   alpha = 0.45, # Discount Factor
                  Nr_Batch = 2**5, # Batch Number for the computation of the value and action function
                   Nr_measures = 10, # The number of different measures
                   Nr_MC = 32, # Number of Monte-Carlo Simmulations for the expectation                   learning_rate_v = 0.001, #The learning rate for the optimization of v
                   learning_rate_a = 0.001, # The learning rate for the optimization of a
                 learning_rate_v = 0.001, 
                   print_training = False, # Whether the training progress should be printed
                   hidden_layers  = 2,
                   nr_neurons = 128,
                transfer_learning = False):
        
        # Fix some parameters (which are not given as variables):         
        self.D = len(Returns_train) # Number of assets
        self.m = len(Returns_train[0][0]) # M = Number of return considered
        self.N = len(Returns_train[0])-1 # Length of the Time Series (minus last entry, which is not considered for preductions)
        
        #Assign the rest of the parameters:
        self.C = C
        self.sigma = sigma
        self.uncertainty = uncertainty
        self.Returns_train = tf.constant(Returns_train)
        self.Returns_1dim = self.Returns_train[:,:,-1]
        self.Returns_train = self.Returns_train[:,:-1,:] # Delete last entry from Returns 
        self.epsilon = epsilon
        self.alpha = alpha
        self.Nr_Batch = Nr_Batch
        self.Nr_measures = Nr_measures
        self.Nr_MC = Nr_MC
        self.learning_rate_v = learning_rate_v
        self.learning_rate_a =learning_rate_a
        self.depth = hidden_layers# Depth of the neural network
        self.nr_neurons = nr_neurons # Number of Neurons of the neural network (same in each layer)
        self.transfer_learning = transfer_learning
        
        #Initialize objects for optimization
        self.optimizer_v = tf.keras.optimizers.Adam(learning_rate_v, beta_1=0.9, beta_2=0.999)
        self.optimizer_a = tf.keras.optimizers.Adam(learning_rate_a, beta_1=0.9, beta_2=0.999)
        
        # Build the Models
        self.v = self.build_model_v(self.depth,self.nr_neurons)
        self.a = self.build_model_a(self.depth,self.nr_neurons)
        self.old_v = self.v   
        
        # Special case epsilon = 0
        if self.epsilon == 0:
            self.Nr_measures = 1
        
    # Create the neural network v
    def build_model_v(self,depth,nr_neurons):  
        """
        Function that creates the neural network for the value function V
        """
        #Input Layer
        x = keras.Input(shape=(self.D,self.m),name = "x")
        #Flatten the dimensions
        v = layers.Flatten()(x)
        # Batch Normalization applied to the input
        v = layers.BatchNormalization()(v)
        # Create the NN       
        v = layers.Dense(nr_neurons,activation = "relu")(v)
        # Create deep layers
        for i in range(depth):
            v = layers.Dense(nr_neurons,activation = "relu")(v)
        # Output Layers
        value_out = layers.Dense(1)(v)
        model = keras.Model(inputs=[x],outputs = [value_out])
        return model
    
    
    # Create the neural network a
    def build_model_a(self,depth,nr_neurons):
        """
        Function that creates the neural network for the action function a
        """
        #Input Layer
        x = keras.Input(shape=(self.D,self.m),name = "x")
        #Flatten the dimensions
        a = layers.Flatten()(x)
        # Batch Normalization applied to the input
        a = layers.BatchNormalization()(a)
        # Create the NN
        a = layers.Dense(nr_neurons,activation = "relu")(a)
        # Create deep layers
        for i in range(depth):
            a = layers.Dense(nr_neurons,activation = "relu")(a) 
        a = layers.Dense(self.D)(a) #,activation = "tanh")(a) #Compute output (restricted to -1,1)
        value_out = layers.Lambda(lambda y: self.C*tf.nn.tanh(y))(a) #multiply output with factor C
        model = keras.Model(inputs=[x],outputs = [value_out])
        return model
    
            
    # Function to sample a batch of states, assumed to be normally distributed
    def generate_batch(self):
        while True:
            r = random.choices(np.arange(self.N),k=self.Nr_Batch) # With bootstrapping from training returns
            returns = tf.gather(self.Returns_train, r, axis = 1) #(D,Batch,m)
            yield tf.transpose(returns,[1,0,2])
            #assets_returns = tf.random.normal((self.Nr_Batch,self.D,self.m), mean=0.0,stddev=self.sigma)
            #yield tf.constant(assets_returns) 
            
    
    
    #Function to sample the next states X1, given current statex according to p
    def generate_X1(self,x): 
        if self.uncertainty == "Wasserstein":
            # Determine the probabilities
            probs = self.p(x) # Shape (Batchsize, N)
            p_hat = tfd.Categorical(probs=probs)
            # Sample according to these probabilities, Shift by 1 to obtain the next return
            closest_returns_indices = np.array(p_hat.sample()+1,dtype = int)      #(Batch)
            next_return = tf.gather(self.Returns_1dim, closest_returns_indices, axis = 1) #(D,Batch)
            # Reshape everything
            next_return = tf.transpose(next_return)
            next_return = tf.expand_dims(next_return,2) #(Batch,D,1)
            #Add noise and ensure that the perurbed return remains in an epsilon ball
            noise_normal = tf.random.normal((self.Nr_Batch,self.D,1), mean=0.0,stddev=self.sigma)
            noise_unif = tf.random.uniform(shape=(self.Nr_Batch,self.D,1),maxval = self.epsilon)
            noise_normal_norm = tf.norm(noise_normal,axis = 1,keepdims = True)
            noise_normal_norm = tf.tile(noise_normal_norm,[1,self.D,1])
            noise = noise_unif*(noise_normal/noise_normal_norm)
            next_return = next_return + noise
            # glue new return with old x (without first return)
            next_return_glued = tf.concat([x[:,:,1:],next_return],axis = 2)
            return next_return_glued #Size: (Batch,D,m)

        elif self.uncertainty == "Parametric":
            # Compute mu
            mean = tf.reduce_mean(x,2) # Size: (Batch,D)
            # Add Noise to the mean
            noise_normal = tf.random.normal((self.Nr_Batch,self.D), mean=0.0,stddev=self.sigma)
            noise_unif = tf.random.uniform(shape=(self.Nr_Batch,self.D),maxval = self.epsilon)
            noise_normal_norm = tf.norm(noise_normal,axis = 1,keepdims = True)
            noise_normal_norm = tf.tile(noise_normal_norm,[1,self.D])
            noise = noise_unif*(noise_normal/noise_normal_norm) # Size: (Batch,D)
            mu = mean + noise # Size: (Batch,D)
            # Compute Sigma
            noise_normal = tf.random.normal((self.Nr_Batch,self.D,self.m), mean=0.0,stddev=self.sigma)
            noise_unif = tf.random.uniform(shape=(self.Nr_Batch,self.D,self.m),maxval = self.epsilon)
            noise_normal_norm = tf.norm(noise_normal,axis = (1,2),keepdims = True)
            noise_normal_norm = tf.tile(noise_normal_norm,[1,self.D,self.m])
            noise = noise_unif*(noise_normal/noise_normal_norm)
            y = x+noise # Size (Batch,D,m)
            mean_y = tf.tile(tf.expand_dims(tf.reduce_mean(x,2),2),[1,1,self.m])
            Sigma = (1/(self.m-1))*tf.linalg.matmul(y-mean_y,y-mean_y,transpose_b=True) # (Batch,D,D)
            # Sample the next value
            distribution = tfp.distributions.MultivariateNormalTriL(loc=mu,
                                                                    scale_tril=tf.linalg.cholesky(Sigma))
            next_return = distribution.sample(event_shape=(self.D))
            next_return = tf.expand_dims(next_return,2)
            next_return_glued = tf.concat([x[:,:,1:],next_return],axis = 2)
            return next_return_glued #Size: (Batch,D,m)
    
    #Helper Function to compute the probabilities in the Wasserstein-Case
    def p(self, x, epsilon=1e-20):        # x has the shape (Batchsize,d,m)
        x = tf.expand_dims(x,1)
        x = tf.tile(x,[1,self.N,1,1]) #((Batchsize,N,d,m))
        y = tf.cast(tf.transpose(self.Returns_train,(1,0,2)),dtype = tf.float32) # Shape (N,d,m)
        y = tf.expand_dims(y,0)   
        y = tf.tile(y,[self.Nr_Batch,1,1,1]) #((Batchsize,N,d,m))
        probs = 1/(tf.sqrt(tf.reduce_sum(tf.square(x-y),axis = (2,3)))+epsilon)  #((Batchsize,N))
        probs_sum = tf.reduce_sum(probs,axis = 1,keepdims = True)
        probs_sum = tf.tile(probs_sum,[1,self.N])
        probs = probs/probs_sum
        return probs # Shape (Batchsize, N)

    def maximize_a(self,X_0):
        A = self.a(X_0,training = True) # Size (Batch,D)   
        #Generate X_1
        X_1 = tf.stack([self.generate_X1(X_0) for _ in range(self.Nr_measures*self.Nr_MC)]) #(Nr_measures*Nr_MC,Batch,D,m) 
        #_1 = tf.map_fn(fn = lambda y: self.generate_X1(X_0), elems = tf.ones(self.Nr_measures*self.Nr_MC)) #(Nr_measures*Nr_MC,Batch,D,m) 
        # Reshape X_1
        X_1_Matrix = tf.reshape(X_1,[self.Nr_measures,self.Nr_MC,self.Nr_Batch,self.D,self.m])
        X_1_Matrix = tf.transpose(X_1_Matrix,[2,3,4,0,1]) #Size: (Batch,D,m,Nr_measures,Nr_MC)
        # Compute c
        A_Matrix = tf.tile(tf.expand_dims(tf.expand_dims(A,2),3),[1,1,self.Nr_measures,self.Nr_MC]) # Size (Batch,D,Nr_measures,Nr_MC)
        C = tf.reduce_sum(X_1_Matrix[:,:,-1,:,:]*A_Matrix,1) # Size: (Batch,Nr_measures,Nr_MC)
        #C = tf.stack([tf.reduce_sum(X_1[i,:,:,-1]*A,1) for i in range(self.Nr_measures*self.Nr_MC)])
        #C = tf.transpose(C)
        #C = tf.reshape(C,[self.Nr_Batch,self.Nr_measures,self.Nr_MC])
        # Compute V
        V_Matrix =  tf.stack([self.old_v(X_1[i,:,:,:],training = False) for i in range(self.Nr_measures*self.Nr_MC)])
        #V_Matrix = tf.map_fn(fn = lambda y: self.v(y,training = False) , elems = X_1) #(Nr_measures*Nr_MC,Batch)    
        V_Matrix = tf.reshape(V_Matrix,[self.Nr_measures,self.Nr_MC,self.Nr_Batch])
        V = tf.transpose(V_Matrix,[2,0,1])
        # Mean w.r.t. Monte Carlo Samples
        means = tf.reduce_mean(C+self.alpha*V,2) # Size: (Batch,Nr_measures,Nr_MC)
        # Minimum w.r.t. the measures
        TV = tf.reduce_min(means,1)
        # Maximize w.r.t. a
        return tf.reduce_mean(-TV)
    
    
    def minimize_quadratic_error(self,X_0):
        A = self.a(X_0,training = False) # Size (Batch,D)      
        #Generate X_1
        #X_1 = tf.map_fn(fn = lambda y: self.generate_X1(X_0), elems = tf.ones(self.Nr_measures*self.Nr_MC)) #(Nr_measures*Nr_MC,Batch,D,m) 
        X_1 = tf.stack([self.generate_X1(X_0) for _ in range(self.Nr_measures*self.Nr_MC)]) #(Nr_measures*Nr_MC,Batch,D,m) 
        # Reshape X_1
        X_1_Matrix = tf.reshape(X_1,[self.Nr_measures,self.Nr_MC,self.Nr_Batch,self.D,self.m])
        X_1_Matrix = tf.transpose(X_1_Matrix,[2,3,4,0,1]) #Size: (Batch,D,m,Nr_measures,Nr_MC)
        # Compute c
        A_Matrix = tf.tile(tf.expand_dims(tf.expand_dims(A,2),3),[1,1,self.Nr_measures,self.Nr_MC]) # Size (Batch,D,Nr_measures,Nr_MC)
        C = tf.reduce_sum(X_1_Matrix[:,:,-1,:,:]*A_Matrix,1) # Size: (Batch,Nr_measures,Nr_MC)
        #C = tf.stack([tf.reduce_sum(X_1[i,:,:,-1]*A,1) for i in range(self.Nr_measures*self.Nr_MC)])
        #C = tf.transpose(C)
        #C = tf.reshape(C,[self.Nr_Batch,self.Nr_measures,self.Nr_MC])
        # Compute V
        V_Matrix =  tf.stack([self.old_v(X_1[i,:,:,:],training = False) for i in range(self.Nr_measures*self.Nr_MC)])
        #V_Matrix = tf.map_fn(fn = lambda y: self.old_v(y,training = False) , elems = X_1) #(Nr_measures*Nr_MC,Batch)    
        V_Matrix = tf.reshape(V_Matrix,[self.Nr_measures,self.Nr_MC,self.Nr_Batch])
        V = tf.transpose(V_Matrix,[2,0,1]) #Size: (Batch,Nr_measures,Nr_MC)
        # Mean w.r.t. Monte Carlo Samples
        means = tf.reduce_mean(C+self.alpha*V,2) # Size: (Batch,Nr_measures)
        # Minimum w.r.t. the measures
        TV = tf.reduce_min(means,1)        
        # Minimize the quadratic difference
        return tf.reduce_mean(tf.square(self.v(X_0,training = True)-TV))

    def grad_optimization_v(self,X_0):
        """
        The gradient of the first optimization step
        """
        with tf.GradientTape() as tape:
            loss_v = self.minimize_quadratic_error(X_0)
        return loss_v, tape.gradient(loss_v ,self.v.trainable_variables)
    
    def grad_optimization_a(self,X_0):
        """
        The gradient of the second optimization step
        """
        with tf.GradientTape() as tape:
            loss_a = self.maximize_a(X_0)
        return loss_a , tape.gradient(loss_a ,self.a.trainable_variables)
        
    
    # The Training routine
    def train(self,Epochs = 10,iterations_a=5,iterations_v=5):
        for Epoch in tqdm(range(Epochs)):
            print("\n#######\nEpoch {} \n#######\n".format(Epoch+1))
            # Transfer weights from v to old_v
            self.old_v = tf.keras.models.clone_model(self.v)
            self.old_v.set_weights(self.v.get_weights())
            self.old_v.trainable = False
            
            #Optimize v
            for _ in range(iterations_v):
                #First generate a batch of states
                X_0 = next(self.generate_batch()) # Size: (Batch,D,m)
                # Minimize the difference between v and TV
                loss_value_v_most_recent, grads_v = self.grad_optimization_v(X_0)
                self.optimizer_v.apply_gradients(zip(grads_v, self.v.trainable_variables))
                print("V: {}".format(loss_value_v_most_recent))
                
            
            #Optimize a
            for _ in range(iterations_a):
                # Transfer Learning, Transfers weights from v to a
                if self.transfer_learning:
                    for k in range(len(self.a.layers[:-4])):
                        self.a.layers[k].set_weights(self.v.layers[k].get_weights())
                        self.a.layers[k].trainable = False
                #First generate a batch of states
                X_0 = next(self.generate_batch()) # Size: (Batch,D,m)
                #Optimize a
                loss_value_a, grads_a = self.grad_optimization_a(X_0)
                self.optimizer_a.apply_gradients(zip(grads_a, self.a.trainable_variables))
                print("a: {}".format(-loss_value_a.numpy()))




                
