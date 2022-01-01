import sys
import pandas as pd
import warnings
import pickle
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics


warnings.simplefilter("ignore", category=ConvergenceWarning)


def create_model(x_train, y_train, pipeline, parameters):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=3)
    grid_search.fit(x_train, y_train)

    return grid_search

def train_model(dataset, pipeline, parameters):
    print(">>>Generanting test/train split")
    docs_train, docs_test, y_train, y_test = train_test_split(dataset['lyric'], dataset['emotion'], test_size=0.2, random_state=None)

    grid_search = create_model(docs_train, y_train, pipeline, parameters)

    # settings for all the candidates explored by grid search.
    n_candidates = len(grid_search.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                 % (grid_search.cv_results_['params'][i],
                    grid_search.cv_results_['mean_test_score'][i],
                    grid_search.cv_results_['std_test_score'][i]))

    # named y_predicted
    y_predicted = grid_search.predict(docs_test)

    # Print the classification report
    print(">>>Printing results")
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=['neg', 'pos']))

    # Print and plot the confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predicted, labels=['neg', 'pos']).ravel()
    acc = (tp.item() + tn.item()) / (tp.item() + tn.item() + fp.item() + fn.item())
    print(">tp: " + str(tp.item()))
    print(">fp: " + str(fp.item()))
    print(">tn: " + str(tn.item()))
    print(">fn: " + str(fn.item()))
    print(">test acc : " + str(acc))

    return grid_search


if __name__ == "__main__":
    # the training data folder must be passed as first argument
    print(">>>Loading dataset")
    dataset = pd.read_csv("other_scripts/lyrics_analysis_Clean.csv")
    dataset['emotion'] = dataset["compound"].apply(lambda x: "pos" if x >= 0 else "neg" )

    sys.stdout = open("test.txt", "w")

    # Pipeline for applying gridseach later
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000, max_iter=9999999, dual=True)),
    ])

    # Fit the pipeline on the training set using grid search for the parameters  
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'vect__use_idf': (True, False),
        'vect__norm': ('l1', 'l2'),
        'vect__sublinear_tf': (True, False),
        'vect__smooth_idf': (True, False),
        'clf__loss': ('hinge', 'squared_hinge'),
        'clf__dual': (True, False),
    }

    print(">>>Applying grid seach")
    model = train_model(dataset, pipeline, parameters)
    with open("modelo.pkl", 'wb') as file:
        pickle.dump(model, file)

    sys.stdout.close()
