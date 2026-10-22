import math

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, mean_squared_error as mse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sympy.physics.quantum.circuitplot import matplotlib


class TwoStepTechnique(BaseEstimator, ABC):
    
    def __init__(self):
        self.classifier=None
        ## self.scaler=None
    
    @abstractmethod
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray]:
        
        pass
    
    @abstractmethod
    def step2(self, X, n, p, u)->BaseEstimator:

        pass
    
    def fit(self, X, s):
        ## self.scaler = preprocessing.StandardScaler().fit(np.copy(X))
        ## X_scaled = self.scaler.transform(np.copy(X))
        results_step1 = self.step1(X,s)
        self.classifier = self.step2(X, results_step1[0],results_step1[1],results_step1[2])
        return self
    
    def predict(self,X):
        ## X_scaled = self.scaler.transform(np.copy(X))
        prediction = self.classifier.predict(X)
        return prediction
    
    def predict_proba(self, X):
        ## X_scaled = self.scaler.transform(np.copy(X))
        probability = self.classifier.predict_proba(X)
        return probability

class SEM(TwoStepTechnique):
    
    def __init__(self,
                 tol = 1.0e-10,
                 max_iter = 100,
                 spy_prop = 0.1,
                 l = 0.15,
                 classifier = LogisticRegression(),
                 seed=331
                ):
        
        super().__init__()
        
        # instantiate the parameters
        ## ADD YOUR CODE HERE
        self.final_classifier = None
        self.tol = tol
        self.max_iter = max_iter
        self.spy_prop = spy_prop
        self.l = l
        self.classifier = classifier
        self.seed = seed
        
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        np.random.seed(self.seed)
        
        # Split the dataset into P (Positive) and M (Mix of positives and negatives)
        P = []
        M = []
        index = 0
        while index < len(s):
            if s[index] == 1:
                P.append(X[index])
            else:
                M.append(X[index])
            index += 1
        
        # Select (randomly) the spies S
        size_S = round(len(P) * self.spy_prop)
        S_index = np.random.choice(range(len(P)), size_S, False)
        S = []
        PRed = []
        index_P = 0
        while index_P in range(len(P)):
            if index_P in S_index:
                S.append(P[index_P])
            else:
                PRed.append(P[index_P])
            index_P += 1
        
        # Update P and MS
        # Updated P becomes PRed (reduced)
        MS = S + M
        
        ### I-EM Algorithm
        PRedMS = PRed + MS
        PRedMSy = [1]*len(PRed) + [0]*len(MS)

        # Train the classifier using P and MS:
        self.classifier.fit(PRedMS, PRedMSy)
        
        # Save the model's score ''score_variation'' using model.score function
        classifiers_scores = [self.classifier.score(PRedMS, PRedMSy)]
        
        # Initialize iterations to 0 and the score variation
        n_iter = 0
        score_variation = classifiers_scores[0]

        # Creating expanded lists that contain every example of MS twice
        # (one positive and one negative, preparation for while loop)
        expanded_PRedMS = []
        expanded_PRedMS += PRed
        for i in MS:
            expanded_PRedMS.append(i)
            expanded_PRedMS.append(i)
        expanded_PRedMSy = [1]*len(PRed) + [1,0]*len(MS)
        
        # Loop while classifier parameters change, i.e. until the score variation is >= tolerance
        while score_variation >= self.tol and n_iter < self.max_iter:
            

            # Expectation step
            expected_probs = list(self.classifier.predict_proba(PRedMS))
            expanded_expected_probs = []
            
            # Create the new training set with the probabilistic labels (weights)
            # Every example from PRed is forced back to weight 1 for the positive example
            # (there is no negative example since its weight would be 0)
            for i in range(len(PRed)):
                expanded_expected_probs.append(float(1))
                expected_probs.pop(0)

            # Every example from MS which is duplicated needs to have its weights adjusted,
            # once for the positive example and for the negative one
            for i in PRedMS[len(PRed):]:
                expanded_expected_probs.append(expected_probs[0][1])
                expanded_expected_probs.append(expected_probs[0][0])
                expected_probs.pop(0)
            
            # Maximization step
            self.classifier.fit(expanded_PRedMS, expanded_PRedMSy, expanded_expected_probs)
            
            # Update score variation and the old score
            classifiers_scores.append(
                self.classifier.score(expanded_PRedMS, expanded_PRedMSy, expanded_expected_probs))
            score_variation = abs(
                classifiers_scores[len(classifiers_scores) - 1] - classifiers_scores[len(classifiers_scores) - 2])
            n_iter += 1

            
        # Print the number of iterations as sanity check
        # print("Number of iterations first step:", n_iter)
        # matplotlib.pyplot.plot(classifiers_scores)
        # matplotlib.pyplot.show()
        
        # Select the threshold t such that l% of spies' probabilities to be positive is below t
        quantity_l = round(len(S) * self.l)
        S_probs = []
        for i in self.classifier.predict_proba(S):
            S_probs.append(i[1])
        S_probs.sort()
        t = S_probs[quantity_l]
        newP = []
        newP += PRed

        # Create N and U
        N = []
        U = []
        # Every point in M that has a probability of being positive that is lower than t
        # gets added to N, otherwise it is added to U
        for point in M:
            proba = self.classifier.predict_proba([point])
            if proba[0][1] < t:
                N.append(point)
            else:
                U.append(point)

        for point in S:
            newP.append(point)
        
        # Return P, N, U
        # print("P:", len(newP))
        # print("N:", len(N))
        # print("U:", len(U))
        return np.array(newP), np.array(N), np.array(U)
        
    
    def step2(self, X, P, N, U)->BaseEstimator:
        np.random.seed(self.seed)

        # Assign every document in the positive set pos the fixed class label 1
        yp = []
        for i in range(len(P)):
            yp.append(1)

        # Assign every document in the likely negative set N the initial class label 0
        yn = []
        for i in range(len(N)):
            yn.append(0)

        ###I-EM Algorithm
        Plist = list(P)
        Nlist = list(N)
        Ulist = list(U)
        Mlist = Ulist+Nlist

        # Train classifier using M and P:
        PN = Plist + Nlist
        PNy = yp + yn
        self.classifier.fit(PN, PNy)

        # Compute the metrics for classifier f_i in delta_i to select the best classifier
        delta_i_list = []

        # Initialize iterations to 0, the score variation, and whether the best classifier has been selected or not.
        n_iter = 0
        classifiers_scores = [self.classifier.score(PN, PNy)]
        score_variation = classifiers_scores[0]

        # Creating expanded lists like in step 1
        PUN = Plist+Mlist
        expanded_PUN = []
        expanded_PUN += Plist
        for i in Mlist:
            expanded_PUN.append(i)
            expanded_PUN.append(i)
        expanded_PUNy = [1]*len(Plist) + [1,0]*len(Mlist)

        # Loop until the variation is > than the tolerance
        while score_variation >= self.tol and n_iter < self.max_iter:

            # Update probabilities
            expected_probs = list(self.classifier.predict_proba(PUN))

            # Create the new training set with the probabilistic labels (weights)
            expanded_expected_probs = []

            # Same mechanism as in step 1
            for i in range(len(Plist)):
                expanded_expected_probs.append(float(1))
                expected_probs.pop(0)

            for i in PUN[len(Plist):]:
                expanded_expected_probs.append(expected_probs[0][1])
                expanded_expected_probs.append(expected_probs[0][0])
                expected_probs.pop(0)

            # Maximization step
            new_classifier = self.classifier.__class__()
            new_classifier.fit(expanded_PUN, expanded_PUNy, expanded_expected_probs)

            # Update parameter variation
            classifiers_scores.append(new_classifier.score(expanded_PUN, expanded_PUNy, expanded_expected_probs))
            score_variation = abs(classifiers_scores[len(classifiers_scores)-1] - classifiers_scores[len(classifiers_scores)-2])
            n_iter += 1

            # Select the best classifier classifier: (final_classifier)
            # Compute the probability approximations
            PrMfiXc1 = self.classifier.score(Ulist, [1] * len(Ulist))
            PrPfiXc2Yc1 = self.classifier.score(Plist, [0] * len(Plist))
            PrMfi1Xc1 = new_classifier.score(Ulist, [1] * len(Ulist))
            PrCfi1Xc2Yc1 = new_classifier.score(Plist, [0] * len(Plist))
            # Compute delta_i, if it is zero or bigger, the previous classifier gets selected
            delta_i = PrMfi1Xc1 - PrMfiXc1 + 2 * (PrCfi1Xc2Yc1 - PrPfiXc2Yc1) * PrMfiXc1
            delta_i_list.append(delta_i)
            if delta_i >= 0:
                self.final_classifier = self.classifier
                break
            else:
                self.final_classifier = new_classifier
                self.classifier = new_classifier



        # print("Number of iterations second step:", n_iter)
        ## matplotlib.pyplot.plot(classifiers_scores)
        ## matplotlib.pyplot.show()
        ## matplotlib.pyplot.plot(delta_i_list)
        ## matplotlib.pyplot.show()


        return self.final_classifier



#--------------------#--------------------#--------------------#--------------------
#-------------------- Second PU Learning Method #--------------------



class ModifiedLogisticRegression(BaseEstimator):

    def __init__(self,
                 tol = 1.0e-10,
                 max_iter = 100,
                 l_rate = 0.01,
                 c = 0,
                 seed = 331,
                 b = None,
                 w = None):

        #instantiate the parameters
        ## ADD YOUR CODE HERE
        self.tol = tol
        self.max_iter = max_iter
        self.l_rate = l_rate
        self.c = c
        self.seed = seed
        self.b = b
        self.w = w

    def log_likelihood(self, x, y):
        #If you use the gradient ascent technique, fill in this part with the log_likelihood function
        #If you prefer to use a different technique, you can leave this empty
        loglik = 0
        i = 0
        while i in range(len(x)):
            loglik += y[i]*np.log(1/(1+self.b**2 + np.exp((np.dot(-self.w,x[i]))))) +\
                      (1-y[i])*np.log(1 - 1/(1+self.b**2 + np.exp((np.dot(-self.w,x[i])))))
            i += 1

        return loglik

    def parameters_update(self, x, y):
        #If you use the gradient ascent technique, fill in this part with the parameter update (both w and b)
        #If you prefer to use a different technique, you can leave this empty
        dloglikdw = 0
        dloglikdb = 0
        i = 0
        while i in range(len(x)):
            exp = np.exp(-self.w.dot(x[i]))
            bexp = self.b ** 2 + exp

            #calculate w shift
            first_term = y[i]/bexp
            second_term = 1/((1+bexp)*bexp)
            dloglikdw += x[i]*exp*(first_term-second_term)
            ## dloglikdw += x[i]*np.exp(-self.w.dot(x[i]))*(y[i]/(self.b**2 + np.exp(-self.w.dot(x[i]))) - 1/((1+self.b**2+np.exp(-self.w.dot(x[i])))*(self.b**2+np.exp(-self.w.dot(x[i])))))

            #calculate b shift
            numerator = 1-y[i]*(1+bexp)
            denumerator = (1+bexp)*bexp
            dloglikdb += 2*self.b*(numerator/denumerator)
            ## dloglikdb += 2*self.b*((1-y[i]*(1+self.b**2+np.exp(-self.w.dot(x[i]))))/((1+self.b**2+np.exp(-self.w.dot(x[i])))*(self.b**2+np.exp(-self.w.dot(x[i])))))
            i += 1

        self.w =  self.w + self.l_rate*dloglikdw
        self.b =  self.b + self.l_rate*dloglikdb

        return

    def fit(self, X, s):
        np.random.seed(self.seed)

        # Initialize w and b
        if self.b == None:
            self.b = 1
            # self.b = np.random.random(1)[0]
        if self.w == None:
            # self.w = np.random.random(len(X[0]))
            self.w = np.array([1]*len(X[0]))
            self.w = self.w/len(self.w)

        # Inizialize the score (log_likelihood), the number of iterations and the score variation.
        scores = [0]
        score_variations = scores
        score_variation = math.inf
        n_iter = 0
        b_list = [self.b]
        w_list = [np.absolute(self.w)]

        #loop until the score variation is lower than tolerance or max_iter is reached
        #while score_variation >= self.tol and n_iter < self.max_iter:
        while score_variation >= self.tol and n_iter < self.max_iter:


            # Update the parameters
            self.parameters_update(X,s)
            b_list.append(self.b)
            w_list.append(np.absolute(self.w))



            # Compute log_likelihood
            score = self.log_likelihood(X,s)

            # Update scores
            scores.append(score)
            score_variations.append(abs(scores[len(scores)-1] - scores[len(scores)-2]))
            score_variation = score_variations[len(score_variations)-1]
            n_iter += 1

        # print("Number of iterations:", n_iter)

        ## matplotlib.pyplot.plot(b_list)
        ## matplotlib.pyplot.show()
        ## matplotlib.pyplot.plot(w_list)
        ## matplotlib.pyplot.show()
        return self

    def estimate_c(self):
        # Estimate the parameter c from b
        ## ADD YOUR CODE HERE
        self.c = 1/(1+self.b**2)
        return self.c

    def predict(self,X):
        ## ADD YOUR CODE HERE
        probabilities = self.predict_proba(X)
        prediction = []
        for prob in probabilities:
            prediction.append(round(prob))

        return prediction

    def predict_proba(self, X):
        ## ADD YOUR CODE HERE
        probability = []
        for x in X:
            ps1x = 1 / (1 + self.b ** 2 + np.exp(-self.w.dot(x)))
            probability.append(ps1x / self.c)

        return probability
    
    
    
    
def remove_from_array(base_array, test_array):
    for index in range(len(base_array)):
        if np.array_equal(base_array[index], test_array):
            base_array.pop(index)
            break