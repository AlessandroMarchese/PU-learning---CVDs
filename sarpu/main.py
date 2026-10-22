# classic imports
import matplotlib.pyplot as plt
from bitarray import bitarray
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR

# sarpu imports
from sarpu.data_generator import *
from sarpu.pu_learning import *
from sarpu.PUmodels import SVCPU, RFPU
from sarpu.labeling_mechanisms import *
from sarpu.data_extending import *
from sarpu.pulib.km.Kernel_MPE_grad_threshold import wrapper as ramaswamy
from sarpu.pulib.tice import tice

# homework imports
from sarpu.pulib.homework import SEM, ModifiedLogisticRegression as MLR

'''
def artificial_attributes(x, y, n_extra_attributes, pdf_method=None):
    # generate the artificial attributes
    extra_atts = []

    prob_dist = calc_probability_distribution(x, y, pdf_method)

    for _ in range(n_extra_attributes):
        att_vals = generate_attributes(x, y, prob_dist=prob_dist, pdf_method=pdf_method)
        extra_atts.append(att_vals)

    return np.stack(extra_atts, axis=1)

def create_data_split(data_path, split):
    # read data into dataframe
    df = pd.read_csv(data_path)

    # contstruct labels
    y = df['HeartDisease'].values

    # construct features
    df = df.drop(columns=["HeartDisease", df.columns[0]])
    x = df.to_numpy()

    # Normalize numerical data with MinMaxScaler
    scaler = MinMaxScaler()
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    x[:, df.columns.get_indexer(numerical_cols)] = scaler.fit_transform(
        x[:, df.columns.get_indexer(numerical_cols)])

    # split into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split)

    return x_train, x_test, y_train, y_test

def add_artificial_attributes(x_train, x_test, y_train, y_test, labelling_model, pdf_method):
    # construct original dataset via concatenation
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    # add artificial attributes to end of x
    n_attributes = len(labelling_model.propensity_attributes)
    attr = artificial_attributes(x, y, n_attributes, pdf_method)
    x = np.concatenate((x, attr), axis=1)

    # resplit x_train and x_test
    x_train = x[:len(x_train), :]
    x_test  = x[len(x_train):, :]

    # Compute the propensity scores
    labelling_model.fit(x_train, y_train)
    propensity_scores = labelling_model.propensity_scores(x_train)

    # compute positive sample selection
    s_train = _calc_s(y_train, propensity_scores)

    return x_train, x_test, s_train
'''


def experiment1(x_train, x_test, y_train, y_test, s_train, propensity_attributes, pdf_method):
    '''
        Compare the different classification and propensity model types

    :return:
    '''

    # plot setting
    fig, ax = plt.subplots()
    lw = 2

    # 1. Define the models: Logistic regression, Support Vector Machine, Random Forest
    f_models = [LR(), SVC(), RF(), None, SVCPU(max_iter=-1), RFPU()]
    p_models = [None, None, None, None, None, None]
    model_names = ["Golden LR", "Golden SVC", "Golden RF", "SAR-EM LR", "SAR-EM SVC", "SAR-EM RF"]
    golden = [True, True, True, False, False, False]
    color = ["darkred", "darkorange", "black", "red", "orange", "black"]

    # iterate over models
    for i in range(len(f_models)):
        if golden[i]:
            f_models[i].fit(x_train[:, :11], y_train)
            model_score = f_models[i].score(x_test[:, :11], y_test) * 100
            y_pred = f_models[i].predict(x_test[:, :11])
            model_f1 = f1_score(y_test, y_pred)
            linestyle = "--"

        else:
            f_model, p_model, info = pu_learn_sar_em(x_train,
                                                        s_train,
                                                        propensity_attributes,
                                                        classification_attributes=None,
                                                        classification_model=f_models[i],
                                                        propensity_model=p_models[i],
                                                        max_its=500,
                                                        slope_eps=0.0001,
                                                        ll_eps=0.0001,
                                                        convergence_window=10,
                                                        refit_classifier=True
                                                        )

            model_score = f_model.model.score(x_test, y_test) * 100
            y_pred = f_model.model.predict(x_test)
            model_f1 = f1_score(y_test, y_pred)
            linestyle = "-"

        # print scores
        print("Model: ", model_names[i])
        print("    - Acc:  %4.2f" % model_score)
        print("    - F1:   %4.2f" % model_f1)

        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_mod = auc(fpr, tpr)

        label = "%s ROC (area = %0.2f)" % (model_names[i], auc_mod)
        ax.plot(
            fpr,
            tpr,
            color=color[i],
            linestyle=linestyle,
            lw=lw,
            label=label,
        )

    # plot random guesser
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Experiment 1: ROC curve")
    plt.legend(loc="lower right")
    plt.show()


def experiment2(x_train, x_test, y_train, y_test, s_train, propensity_attributes, pdf_method):
    '''
        Compare the different classification and propensity model types

    :return:
    '''

    # plot setting

    lw = 2

    # 1. Define the models: Logistic regression, Support Vector Machine, Random Forest
    f_models = [None, SVCPU(max_iter=-1), RFPU()]
    p_models = [None, SVCPU(max_iter=-1), RFPU()]
    f_model_names = ["SAR-EM LR", "SAR-EM SVC", "SAR-EM RF"]
    p_model_names = ["LR", "SVC", "RF"]
    color = ["darkred", "darkorange", "black"]

    # iterate over models
    for i in range(len(f_models)):
        fig, ax = plt.subplots()
        for j in range(len(p_models)):
            f_model, p_model, info = pu_learn_sar_em(x_train,
                                                        s_train,
                                                        propensity_attributes,
                                                        classification_attributes=None,
                                                        classification_model=f_models[i],
                                                        propensity_model=p_models[j],
                                                        max_its=500,
                                                        slope_eps=0.0001,
                                                        ll_eps=0.0001,
                                                        convergence_window=10,
                                                        refit_classifier=True
                                                        )

            model_score = f_model.model.score(x_test, y_test) * 100
            y_pred = f_model.model.predict(x_test)
            model_f1 = f1_score(y_test, y_pred)
            linestyle = "-"

            # print scores
            print("Classification Model: ", f_model_names[i])
            print("Propensity Model: ", p_model_names[j])
            print("    - Acc:  %4.2f" % model_score)
            print("    - F1:   %4.2f" % model_f1)

            # ROC-AUC
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_mod = auc(fpr, tpr)

            label = "p=%s ROC (area = %0.2f)" % (p_model_names[j], auc_mod)
            ax.plot(
                fpr,
                tpr,
                color=color[j],
                linestyle=linestyle,
                lw=lw,
                label=label,
            )

        # plot random guesser
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Experiment 2: ROC curve f=%s" % f_model_names[i])
        plt.legend(loc="lower right")
        plt.show()


def experiment3(x_train, x_test, y_train, y_test, s_train, propensity_attributes, pdf_method):
    '''
        Compare SAR-EM against MLR and S-EM
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param s_train:
    :param propensity_attributes:
    :param pdf_method:
    :return:
    '''

    # matplotlib
    fig, ax = plt.subplots()
    lw = 2
    color = ["red", "orange", "black"]

    # S-EM model
    sem_clf = SEM(tol=1.0e-10, max_iter=100, spy_prop=0.1, l=0.15,
                        classifier=SVC(probability=True), seed=331)
    sem_clf.fit(x_train[:, :11], s_train)

    # MLR model
    max_iter = 100
    l_rate = 1 / max_iter
    mlr_clf = MLR(max_iter=max_iter, l_rate=l_rate, seed=331)
    mlr_clf.fit(x_train[:, :11], s_train)
    mlr_clf.estimate_c()

    # SAR-EM SVC
    f_model, p_model, info = pu_learn_sar_em(x_train,
                                             s_train,
                                             propensity_attributes,
                                             classification_attributes=None,
                                             classification_model=SVCPU(max_iter=-1),
                                             propensity_model=SVCPU(max_iter=-1),
                                             max_its=500,
                                             slope_eps=0.0001,
                                             ll_eps=0.0001,
                                             convergence_window=10,
                                             refit_classifier=True
                                             )

    # evaluate
    models = [sem_clf, mlr_clf, f_model.model]
    model_names = ["S-EM", "MLR", "SAR-EM SVC"]
    for i in range(len(models)):
        model = models[i]
        if i == 0:
            acc =model.final_classifier.score(x_test[:, :11], y_test)
            pred = model.predict(x_test[:, :11])
        elif i == 1: # no artificial attributes
            pred = model.predict(x_test[:, :11])
            acc = accuracy_score(y_test, pred)
        else:
            acc = model.score(x_test, y_test)
            pred = model.predict(x_test)

        acc = acc * 100
        f1 = f1_score(y_test, pred)

        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, pred)
        auc_mod = auc(fpr, tpr)

        # print scores
        print("Model: ", model_names[i])
        print("    - Acc:  %4.2f" % acc)
        print("    - F1:   %4.2f" % f1)

        label = "%s ROC (area = %0.2f)" % (model_names[i], auc_mod)
        ax.plot(
            fpr,
            tpr,
            color=color[i],
            lw=lw,
            label=label,
        )

    # plot random guesser
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Experiment 3: ROC curve")
    plt.legend(loc="lower right")
    plt.show()


def experiment4(x_train, x_test, y_train, y_test, s_train, propensity_attributes, pdf_method):
    '''
        Compare SAR-EM against SCAR-KM2 and SCAR-TICE
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param s_train:
    :param propensity_attributes:
    :param pdf_method:
    :return:
    '''

    # matplotlib
    fig, ax = plt.subplots()
    lw = 2
    color = ["red", "orange", "black"]

    # SCAR-KM2
    # label frequency estimation
    (_, c_est_km2) = ramaswamy(x_train[:, :11], x_train[s_train == 1, :11])
    # pu learning
    f_model_km2, info = pu_learn_scar_c(x_train[:, :11],
                                    s_train,
                                    c_est_km2,
                                    classification_model=SVCPU(max_iter=-1),
                                    classification_attributes=None)

    # SCAR-TICE
    # label frequency estimation
    tice_folds = np.random.randint(5, size=len(s_train))
    k = 5
    s_train_tice = bitarray(list(s_train == 1))
    (c_est, _) = tice.tice(x_train[:, :11], s_train_tice, k, tice_folds)
    # pu learning
    f_model_tic, info = pu_learn_scar_c(x_train[:, :11],
                                    s_train,
                                    c_est,
                                    classification_model=SVCPU(max_iter=-1),
                                    classification_attributes=None)

    # SAR-EM SVC
    f_model_sarem, p_model, info = pu_learn_sar_em(x_train,
                                                     s_train,
                                                     propensity_attributes,
                                                     classification_attributes=None,
                                                     classification_model=SVCPU(max_iter=-1),
                                                     propensity_model=SVCPU(max_iter=-1),
                                                     max_its=500,
                                                     slope_eps=0.0001,
                                                     ll_eps=0.0001,
                                                     convergence_window=10,
                                                     refit_classifier=True
                                                     )

    # evaluate
    models = [f_model_km2, f_model_tic, f_model_sarem]
    model_names = ["SCAR-KM2", "SCAR-TICE", "SAR-EM SVC"]
    for i in range(len(models)):
        model = models[i]
        if i != 2:
            acc = model.model.score(x_test[:, :11], y_test)
            pred = model.model.predict(x_test[:, :11])
        else:
            acc = model.model.score(x_test, y_test)
            pred = model.model.predict(x_test)

        f1 = f1_score(y_test, pred)

        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, pred)
        auc_mod = auc(fpr, tpr)

        # print scores
        print("Model: ", model_names[i])
        print("    - Acc:  %4.2f" % acc)
        print("    - F1:   %4.2f" % f1)

        label = "%s ROC (area = %0.2f)" % (model_names[i], auc_mod)
        ax.plot(
            fpr,
            tpr,
            color=color[i],
            lw=lw,
            label=label,
        )

    # plot random guesser
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Experiment 4: ROC curve")
    plt.legend(loc="lower right")
    plt.show()


def experiment5(x_train, x_test, y_train, y_test, s_train, labelling_model):
    '''
        Compare SAR-EM probability distributions
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param s_train: selected labelling according to uniform distribution
    :param propensity_attributes:
    :param pdf_method:
    :return:
    '''

    # matplotlib
    fig, ax = plt.subplots()
    lw = 2
    color = ["red", "orange", "black"]

    propensity_attributes = labelling_model.propensity_attributes

    # SAR-EM SVC: UNIFORM DISTRIBUTION
    f_model_uniform, p_model, info = pu_learn_sar_em(x_train,
                                                     s_train,
                                                     propensity_attributes,
                                                     classification_attributes=None,
                                                     classification_model=SVCPU(max_iter=-1),
                                                     propensity_model=SVCPU(max_iter=-1),
                                                     max_its=500,
                                                     slope_eps=0.0001,
                                                     ll_eps=0.0001,
                                                     convergence_window=10,
                                                     refit_classifier=True
                                                     )


    # generate labels according to multinomial distribution
    x_train_multinomial, x_test_multinomial, s_train_multinomial = add_artificial_attributes(x_train[:, :11],
                                                                                             x_test[:, :11],
                                                                                             y_train,
                                                                                             y_test,
                                                                                             labelling_model,
                                                                                             pdf_method="multinomial")

    # SAR-EM SVC: UNIFORM DISTRIBUTION
    f_model_multinomial, p_model, info = pu_learn_sar_em(x_train_multinomial,
                                                     s_train_multinomial,
                                                     propensity_attributes,
                                                     classification_attributes=None,
                                                     classification_model=SVCPU(max_iter=-1),
                                                     propensity_model=SVCPU(max_iter=-1),
                                                     max_its=500,
                                                     slope_eps=0.0001,
                                                     ll_eps=0.0001,
                                                     convergence_window=10,
                                                     refit_classifier=True
                                                     )

    # generate labels according to KDE distribution
    x_train_kde, x_test_kde, s_train_kde = add_artificial_attributes(x_train[:, :11],
                                                                     x_test[:, :11],
                                                                     y_train,
                                                                     y_test,
                                                                     labelling_model,
                                                                     pdf_method="kde")

    # SAR-EM SVC: KDE DISTRIBUTION
    f_model_kde, p_model, info = pu_learn_sar_em(x_train_kde,
                                                         s_train_kde,
                                                         propensity_attributes,
                                                         classification_attributes=None,
                                                         classification_model=SVCPU(max_iter=-1),
                                                         propensity_model=SVCPU(max_iter=-1),
                                                         max_its=500,
                                                         slope_eps=0.0001,
                                                         ll_eps=0.0001,
                                                         convergence_window=10,
                                                         refit_classifier=True
                                                         )

    # Evaluate
    models = [f_model_uniform, f_model_multinomial, f_model_kde]
    model_names = ["UNIFORM", "MULTINOMIAL", "KDE"]
    for i in range(len(models)):
        model = models[i]
        acc = model.model.score(x_test, y_test)
        pred = model.model.predict(x_test)

        f1 = f1_score(y_test, pred)

        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, pred)
        auc_mod = auc(fpr, tpr)

        # print scores
        print("Model: ", model_names[i])
        print("    - Acc:  %4.2f" % acc)
        print("    - F1:   %4.2f" % f1)

        label = "%s ROC (area = %0.2f)" % (model_names[i], auc_mod)
        ax.plot(
            fpr,
            tpr,
            color=color[i],
            lw=lw,
            label=label,
        )

    # plot random guesser
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Experiment 5: ROC curve")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    # settings
    read_ds = True
    write_ds = False
    run_experiments = [False, True, False, False, False]
    data_path = "../heart_processed.csv"
    dataset_folder = "../dataset/"
    split = 0.8
    propensity_attributes = [11, 12, 13, 14]
    propensity_attributes_signs = [1, 1, 1, 1]
    labeling_model_type = "simple_0.2_0.8"

    # 1. data preparation
    labelling_model = parse_labeling_model(labeling_model_type, propensity_attributes, propensity_attributes_signs)
    pdf_method = None
    if not read_ds:
        # create split
        x_train, x_test, y_train, y_test = create_data_split(data_path=data_path, split=split)

        # add artificial attributes

        x_train_attr, x_test_attr, s_train = add_artificial_attributes(x_train, x_test, y_train, y_test, labelling_model, pdf_method)

        # save dataset
        if write_ds:
            write_dataset(x_train_attr, x_test_attr, y_train, y_test, s_train, dataset_folder)
    else:
        x_train_attr, x_test_attr, y_train, y_test, s_train = read_dataset(dataset_folder)

    print("Number of positive training samples: ", int(np.sum(y_train)))
    print("Number of labeled training samples: ", int(np.sum(s_train)))

    if run_experiments[0]:
        ''' 
            EXPERIMENT 1 : SAR-EM overview
        '''
        print("Experiment 1: SAR-EM using different classification model types")
        experiment1(x_train=x_train_attr,
                    x_test=x_test_attr,
                    y_train=y_train,
                    y_test=y_test,
                    s_train=s_train,
                    propensity_attributes=labelling_model.propensity_attributes,
                    pdf_method=pdf_method)
    if run_experiments[1]:
        ''' 
            EXPERIMENT 2 : SAR-EM changing propensity score models
        '''
        print("Experiment 2: SAR-EM using different propensity score model types")
        experiment2(x_train=x_train_attr,
                    x_test=x_test_attr,
                    y_train=y_train,
                    y_test=y_test,
                    s_train=s_train,
                    propensity_attributes=labelling_model.propensity_attributes,
                    pdf_method=pdf_method)

    if run_experiments[2]:
        '''
            EXPERIMENT 3
        '''
        print("Experiment 3: SAR-EM versus S-EM and MLR ")
        experiment3(x_train=x_train_attr,
                    x_test=x_test_attr,
                    y_train=y_train,
                    y_test=y_test,
                    s_train=s_train,
                    propensity_attributes=labelling_model.propensity_attributes,
                    pdf_method=pdf_method)

    if run_experiments[3]:
        '''
            EXPERIMENT 4
        '''
        print("Experiment 4: SAR-EM versus SCAR-KM2 and SCAR-TICE")
        experiment4(x_train=x_train_attr,
                    x_test=x_test_attr,
                    y_train=y_train,
                    y_test=y_test,
                    s_train=s_train,
                    propensity_attributes=labelling_model.propensity_attributes,
                    pdf_method=pdf_method)
    if run_experiments[4]:
        '''
            EXPERIMENT 5
        '''
        print("Experiment 5: SAR-EM variable probability distributions")
        experiment5(x_train=x_train_attr,
                    x_test=x_test_attr,
                    y_train=y_train,
                    y_test=y_test,
                    s_train=s_train,
                    labelling_model=labelling_model)


