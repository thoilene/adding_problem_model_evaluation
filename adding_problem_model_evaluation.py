#adding_problem_model_evaluation.py

# Disable annoying warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# Modules import

import tensorflow as tf

# Disable annoying tf warnings
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import random
import matplotlib.pyplot as plt

from scikeras.wrappers import KerasRegressor
import pickle

from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

# Setting seed to ensure reproducibility of models and results
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0) 
tf.config.experimental.enable_op_determinism()


# class Adding_Problem_DataGenerator 

class Adding_Problem_DataGenerator():
    """
    class Adding_Problem_DataGenerator is used to generate data for adding problem. It creates singles sequences and builds
    datasets which are then used for model development and evaluation.  
    """

    def __init__(self, seq_len=50, train_len=10000, test_len=2000, valid_len=1000):
        
        """
        __init__ initialize each instance of this class.

        :param seq_len: length of the sequences to be generated - default value is 50
        :param train_len: number of samples for training set - default value is 10000
        :param test_len: number of samples for test set - default value is 2000
        :param valid_len: number of samples for validation set - default value is 1000
        
        :return: self - instance of this class
        """ 
        
        self.seq_len = seq_len
        self.train_len = train_len
        self.test_len = test_len
        self.valid_len = valid_len

    def generate_adding_problem_seq(self, data_len, random_state=42):
        """ 
        generate_adding_problem_seq generates a dataset of sequences for adding problem
        
        :param data_len: number of samples for the dataset to be created
        :param random_state: numpy random seed to be set before generating random numbers - defualt value is 42 
        
        :return X, Y: X is the set of samples of sequences. Y is the set of correnponding targets.
        """
        
        np.random.seed(random_state)

        X_num = np.random.uniform(low=0, high=1, size=(data_len, self.seq_len, 1))
        X_mask = np.zeros((data_len, self.seq_len, 1))
        Y = np.ones((data_len, 1))
        for i in range(data_len):
            positions = np.random.choice(self.seq_len, size=2, replace=False)
            X_mask[i, positions] = 1
            Y[i, 0] = np.sum(X_num[i, positions])
        X = np.append(X_num, X_mask, axis=2)
        return X, Y

    def generate_model_data(self):
        """
        generate_model_data generates train dataset, test dataset and validation dataset for initial parameters
        
        :return X_train, y_train, X_test, y_test, X_val, y_val
            X_train is the set of samples of sequences for train dataset
            y_train is the set of correnponding targets for train dataset
            X_test is the set of samples of sequences for test dataset
            y_test is the set of correnponding targets for test dataset
            X_val is the set of samples of sequences for validation dataset
            y_val is the set of correnponding targets for validation dataset
        """
    
        X_train, y_train = None, None
        X_val, y_val = None, None
        X_test, y_test = None, None
        
        if self.train_len > 1:
            X_train, y_train = self.generate_adding_problem_seq(data_len=self.train_len, random_state=1)
            
        if self.valid_len > 1:
            X_val, y_val = self.generate_adding_problem_seq(data_len=self.valid_len, random_state=2)
            
        if self.test_len > 1:
            X_test, y_test = self.generate_adding_problem_seq(data_len=self.test_len, random_state=3)
            
        return X_train, y_train, X_test, y_test, X_val, y_val
    
######################################################################################################

# Global functions

def load_data(filename):
    """
    load_data pickle-loads data from file 
    :param filename: name of the file from where data should be read in.
    :return data: content of the file 
    """
    # open a file, where you stored the pickled data
    file = open(filename, 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()
    
    return data

# save_data
def save_data(data, filename):
    """
    save_data dumps data into file using pickle.
    :param data: data to be saved (in simple data structure)
    :param filename: name of the file where to store the data  
    """
    #open a file, where you ant to store the data
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)

    # close the file
    file.close()

# create_simple_model    
def create_simple_model(units=100, learning_rate=0.01):
    """
    create_simple_model build a model with SimpleRNN, identity initializer for recurrent initializer and relu activation.
    It uses SGD optimizer (with learning rate as only parameter), Huber as loss function and mean squared error (mse) 
    as performance mesurement metric. 
    
    :param units - number of units for the recurrent layer - default is 100
    :param learning_rate - learning rate for the optimizer - default is 0.01
    
    : return model - compiled keras sequential model
    """
    # Identity initializer
    initializer = tf.keras.initializers.Identity()
    
    # keras sequential model
    model = tf.keras.models.Sequential([
          tf.keras.layers.SimpleRNN(units, input_shape=[None, 2], recurrent_initializer=initializer, activation="relu"),
          tf.keras.layers.Dense(1)
        ])
    # compiling the model using Huber loss function, SGD optimizer and mse as metric
    model.compile(loss=tf.keras.losses.Huber(),
                        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                        metrics=['mse'])
    return model

# create_model
def create_model(units=100, learning_rate=1e-2, momentum=0.9, clipnorm=1.0):
    """
    create_simple_model build a model with SimpleRNN, identity initializer for recurrent initializer and relu activation.
    It uses SGD optimizer (with learning rate, momentum and clipnorm as parameters), Huber as loss function and 
    mean squared error (mse) as performance mesurement metric. 
    
    :param units - number of units for the recurrent layer - default is 100
    :param learning_rate - learning rate for the optimizer - default is 0.01
    :param momentum - momentum for the optimizer - default is 0.9
    :param clipnorm - clipnorm for the optimizer - default is 1.0
    
    : return model - compiled keras sequential model
    """
    # Identity initializer
    initializer = tf.keras.initializers.Identity()
    
    # keras sequential model
    model = tf.keras.models.Sequential([
          tf.keras.layers.SimpleRNN(units, input_shape=[None, 2], recurrent_initializer=initializer, activation="relu"),
          tf.keras.layers.Dense(1)
        ])
    # compiling the model using Huber loss function, SGD optimizer and mse as metric
    model.compile(loss=tf.keras.losses.Huber(),
                        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, clipnorm=clipnorm),
                        metrics=['mse'])
    return model

###############################################################################################

# class Adding_Problem_ModelEvaluator

class Adding_Problem_ModelEvaluator():
    """
    Adding_Problem_ModelEvaluator groups all functionalities to evaluate the model for adding problem.
    These include training the model and storing the learning history as well as tuning the model.
    """

    def __init__(self, sequence_lengths = [50], train_len=10000, test_len=2000, valid_len=1000):
        """
        __init__ initialize each instance of this class.

        :param sequence_lengths: list of the length of sequences to be generated - default value is [50]
        :param train_len: number of samples for training set - default value is 10000
        :param test_len: number of samples for test set - default value is 2000
        :param valid_len: number of samples for validation set - default value is 1000
        
        :return: self - instance of this class
        """ 
        self.sequence_lengths = sequence_lengths 
        self.seq_len_evaluation_results = {} 
        self.lr_len_evaluation_results = {}
        self.tuning_results = {}
        self.train_len=train_len
        self.test_len=test_len
        self.valid_len=valid_len

    # run_evaluation_for_sequence_len
    
    def run_evaluation_for_sequence_len(self, epochs=20, lrs={}, verbose=0, save=False):
        """
        run_evaluation_for_sequence_len creates simple models corresponding the the sequence lengths of the class instance
        using the function create_simple_model, then trains the models and save the learning history of each of them.

        :param epochs - number of learning steps of the model - default value is 20
        :param lrs: dictionnary of learning rates corresponding to sequence lengths of the class instance - default is {} (empty dict)
        :param verbose: verbosity of the training process. 0 mean no output - 0 or 2 mean some output.
        :param save: flag indicating whether the results (learning history) of the model training should be saved into a file
        
        :return: evaluation_results - dictionnary of learning histories
        """ 
        evaluation_results = {}

        for seq_len in self.sequence_lengths:
            
            lr = 0.01
            if seq_len in lrs.keys():
                lr = lrs[seq_len]
            # generate_sequence_data
            data_gen = Adding_Problem_DataGenerator(seq_len=seq_len, train_len=self.train_len, 
                                                    test_len=self.test_len, valid_len=self.valid_len )
            X_train, y_train, X_test, y_test, X_val, y_val = data_gen.generate_model_data()

            # create ML model
            model = create_simple_model(learning_rate=lr)
       
            # set tensorflow and numpy random seed for reproductibility
            np.random.seed(0)
            random.seed(0)
            tf.random.set_seed(0)
            tf.keras.utils.set_random_seed(0) 

            print(f"running model fit for sequence length {seq_len} for learning rate {lr}...")
            history = model.fit(X_train, y_train, validation_data=[X_val, y_val], 
                                epochs=epochs, workers=32, use_multiprocessing=True, verbose=verbose, shuffle=False)

            # evaluate model and store results
            evaluation_results[seq_len] = model.evaluate(X_test, y_test), history
            
            # save data if requested
            if save:
                save_data(data=evaluation_results, filename=f"/content/sample_data/data/evaluation_for_seq_{seq_len}.pkl")
        
        self.seq_len_evaluation_results = evaluation_results

        return evaluation_results
    
    # run_tuned_evaluation_for_sequence_len
    
    def run_tuned_evaluation_for_sequence_len(self, epochs=20, verbose=0, lrs={}, momentums={}, clipnorms={}, save=False):
        """
        run_tuned_evaluation_for_sequence_len creates models corresponding the the sequence lengths of the class instance
        using the function create_model, then trains the models and save the learning history of each of them.

        :param epochs - number of learning steps of the model - default value is 20
        :param lrs: dictionnary of learning rates corresponding to sequence lengths of the class instance - default is {} (empty dict)
        :param momentums: dictionnary of momentums corresponding to sequence lengths of the class instance - default is {} (empty dict)
        :param clipnorms: dictionnary of clipnorms corresponding to sequence lengths of the class instance - default is {} (empty dict)
        :param verbose: verbosity of the training process. 0 mean no output - 0 or 2 mean some output.
        :param save: flag indicating whether the results (learning history) of the model training should be saved into a file
        
        :return: evaluation_results - dictionnary of learning histories
        """ 
        evaluation_results = {}

        for seq_len in self.sequence_lengths:
            
            lr = 0.01 # default value for learning rate
            if seq_len in lrs.keys():
                lr = lrs[seq_len]
            momentum = 0.0 # default value for momentum
            if seq_len in momentums.keys():
                momentum = momentums[seq_len]
            clipnorm = 0.0 # default value for clipnorm
            if seq_len in clipnorms.keys():
                clipnorm = clipnorms[seq_len]
                
            # generate_sequence_data
            data_gen = Adding_Problem_DataGenerator(seq_len=seq_len, 
                                                    train_len=self.train_len, 
                                                    test_len=self.test_len, 
                                                    valid_len=self.valid_len )
            X_train, y_train, X_test, y_test, X_val, y_val = data_gen.generate_model_data()

            # create ML model
            #model = create_simple_model(learning_rate=lr)
            model = create_model(units=100, learning_rate=lr, momentum=momentum, clipnorm=clipnorm)
            
            # set tensorflow and numpy random seed for reproductibility
            np.random.seed(0)
            random.seed(0)
            tf.random.set_seed(0)
            tf.keras.utils.set_random_seed(0) 

            print(f"running model fit for sequence length {seq_len} for learning rate {lr}...")
            history = model.fit(X_train, y_train, validation_data=[X_val, y_val], 
                                epochs=epochs, workers=32, use_multiprocessing=True, 
                                verbose=verbose, shuffle=False)

            # evaluate model and store results
            evaluation_results[seq_len] = model.evaluate(X_test, y_test), history
            
            # save data/resluts if requested
            if save:
                save_data(data=evaluation_results, filename=f"/content/sample_data/data/tuned_evaluation_for_seq_{seq_len}.pkl")

        self.seq_len_evaluation_results = evaluation_results

        return evaluation_results
    
    
    # run_evaluation_for_learning_rate
    def run_evaluation_for_learning_rate(self, epochs=100, verbose=0, save= False):
        """
        run_evaluation_for_learning_rate creates models corresponding the the sequence lengths of the class instance
        using the function create_simple_model, then trains the models with learning rate scheduler and save the leaning 
        history of each of them. 

        :param epochs - number of learning steps of the model - default value is 20
        :param verbose: verbosity of the training process. 0 mean no output - 0 or 2 mean some output.
        :param save: flag indicating whether the results (learning history) of the model training should be saved into a file
        
        :return: evaluation_results - dictionnary of learning histories
        """ 
        histories = {}
        for seq_len in self.sequence_lengths:
            
            # generate_sequence_data
            data_gen = Adding_Problem_DataGenerator(seq_len=seq_len)
            X_train, y_train, X_test, y_test, X_val, y_val = data_gen.generate_model_data()

            # create ML model
            model = create_simple_model()
            
            # set tensorflow and numpy random seed for reproductibility
            np.random.seed(0)
            random.seed(0)
            tf.random.set_seed(0)
            tf.keras.utils.set_random_seed(0)
            
            # initializing learning rate scheduler
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 30))

            print(f"evaluation_for_learning_rate: running fit for sequence length {seq_len} ...")
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=[X_val, y_val], 
                                callbacks=[lr_schedule], verbose=verbose, workers=16, use_multiprocessing=True)
            
            histories[seq_len] = history
            
            # save data/resluts if requested
            if save:
                save_data(data=histories, filename=f"/content/sample_data/data/lr_evaluation_for_seq_{seq_len}.pkl")
            
            self.lr_len_evaluation_results = histories

        return histories

    # run_parameter_tuning
    def run_parameter_tuning(self, epochs=100, n_calls=20, n_initial_points=5, verbose=0, cv=3, save=False):
        """
        run_parameter_tuning search for the parameter set with the highest performance in term of 
        negative mean squared error (neg_mean_squared_error) using bayesian optimization with cross validation. 
        
        :param epochs - number of learning steps of the model - default value is 100
        :param n_calls: number of all iterations for parameter tuning - default is 20
        :param n_initial_points: number of iterations where hyperparameter set is chosen randomly - default is 5
        :param verbose: verbosity of the training process. 0 mean no output - 0 or 2 mean some output.
        :param cv: number of splits for cross validation - default is 3
        :param save: flag indicating whether the results (learning history) of the model training should be saved into a file
        
        :return: tuning_results - dictionnary of parameters with the highest performance metric. 
        """
        self.tuning_results = {}
        for seq_len in self.sequence_lengths:
            
            # generate_sequence_data
            
            data_gen = Adding_Problem_DataGenerator(seq_len=seq_len, train_len=self.train_len, 
                                                    test_len=self.test_len, valid_len=self.valid_len)
            X_train, y_train, X_test, y_test, X_val, y_val = data_gen.generate_model_data()
            
            print(f"running the hyper parameter tuning for sequence length {seq_len}...")
            
            # set tensorflow and numpy random seed for reproductibility
            np.random.seed(0)
            random.seed(0)
            tf.random.set_seed(0)
            tf.keras.utils.set_random_seed(0)
            
            # Function to compute the computation of the performance of the model at each iteration
            def cv_score(learning_rate, momentum, clipnorm):
                score = cross_val_score(
                            KerasRegressor(build_fn = create_model, learning_rate=learning_rate, 
                                           momentum=momentum, clipnorm=clipnorm, verbose = 0, epochs=epochs),
                            X_train, y_train, scoring='neg_mean_squared_error', verbose=0, n_jobs=32, cv=3).mean()
                score = np.array(score)
                return score
            
            # sets of hyperparameters for model tuning
            bds = {'learning_rate':[1.e-4, 1.e-2], 'momentum':[0.0, 0.99], 'clipnorm':[0., 100]}
            
            # Initializing bayesian optimizer
            optimizer = BayesianOptimization(f=cv_score, pbounds=bds, random_state=42, verbose=2)
            
            # Runnning bayesian optimization 
            optimizer.maximize(init_points = n_initial_points, n_iter = n_calls-n_initial_points)
            self.tuning_results[seq_len] = optimizer.max
            
            if save:
                save_data(data=optimizer.max, filename=f"/content/sample_data/data/tuning_evaluation_for_seq_{seq_len}.pkl")
            
            print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

        self.tuning_results
        
        