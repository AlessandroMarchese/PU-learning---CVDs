from sklearn.metrics import f1_score

from Homework.PU_Learning import SEM
from Homework.PU_Learning import ModifiedLogisticRegression
import pandas as pd
import dill as pickle
from sklearn.preprocessing import Normalizer, normalize
from sklearn.metrics import f1_score, mean_squared_error as mse
from main import prepare_data
from main import test_with_artificial_propensity_attributes
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sarpu.PUmodels import RFPU
from sarpu.pu_learning import pu_learn_sar_em

data_path = "../heart_processed.csv"

method = "artificial"

if method == "artificial":

    x_train, s_train, propensity_attributes, y_train, x_test, y_test = test_with_artificial_propensity_attributes()

else:
    x_train, s_train, propensity_attributes, y_train, x_test, y_test = test_with_artificial_propensity_attributes()

print("Number of positive training samples: ", int(np.sum(y_train)))
print("Number of labeled training samples: ", int(np.sum(s_train)))

## SEM
classification_model_SEM = SEM(classifier= RandomForestClassifier())
classification_model_SEM.fit(x_train,s_train)

y_pred = classification_model_SEM.predict(np.copy(x_test))

print("SEM ","F1 score:", f1_score(y_test, y_pred))

## MLR
classification_model_MLR = ModifiedLogisticRegression()
classification_model_MLR.fit(x_train,s_train)
classification_model_MLR.estimate_c()

y_pred = classification_model_MLR.predict(np.copy(x_test))

print("MLR ","F1 score:", f1_score(y_test, y_pred))

# SAR-EM with RF
classification_model_rf, propensity_model_rf, info_rf = pu_learn_sar_em(x_train,
                                                                            s_train,
                                                                            propensity_attributes,
                                                                            classification_attributes=None,
                                                                            classification_model=RFPU(),
                                                                            propensity_model=RFPU(),
                                                                            max_its=500,
                                                                            slope_eps=0.0001,
                                                                            ll_eps=0.0001,
                                                                            convergence_window=10,
                                                                            refit_classifier=True
                                                                            )
y_pred = classification_model_rf.model.predict(np.copy(x_test))

print("RF SAR EM ","F1 score:", f1_score(y_test, y_pred))