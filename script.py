import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2


def cross_validation_report(X, y, model):
    kfold = cross_validation.KFold(n=len(X), n_folds=5, random_state=1)
    scores = cross_validation.cross_val_score(
        model,
        X,
        y,
        # scoring='log_loss',
        cv=kfold,
    )
    #
    print(scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def create_submission(model, X_train, y_train, test, filename):
    print('training...')
    model.fit(X_train, y_train)

    print('predicting...')
    print(model.classes_)
    pred = model.predict_proba(test)
    print(pred)
    print('creating submission...')
    submission = pd.DataFrame({'shot_id': test.index.values,
                               'shot_made_flag': pred[:, 1]})
    submission.to_csv(filename, index=False)

    print('file saved at ' + os.path.realpath('.') + filename)


def select_features(X_train, y_train):
    threshold = 0.90
    vt = VarianceThreshold().fit(X_train)
    feat_var_threshold = X_train.columns[vt.variances_ > threshold * (1 - threshold)]
    # print(feat_var_threshold)
    # print(len(feat_var_threshold))

    # Random Forest feature importance
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    feature_imp = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=["importance"])
    # print(feature_imp)
    feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(35).index
    # print(feat_imp_20)

    X_minmax = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_train)
    X_scored = SelectKBest(score_func=chi2, k='all').fit(X_minmax, y_train)
    feature_scoring = pd.DataFrame({
        'feature': X_train.columns,
        'score': X_scored.scores_
    })
    feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(35)['feature'].values
    # print(feat_scored_20)

    rfe = RFE(LogisticRegression(), 20)
    rfe.fit(X_train, y_train)
    feature_rfe_scoring = pd.DataFrame({
        'feature': X_train.columns,
        'score': rfe.ranking_
    })
    feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
    # print(feat_rfe_20)

    features = np.hstack([
        feat_var_threshold,
        feat_imp_20,
        feat_scored_20,
        feat_rfe_20
    ])
    # print(features)
    # features = map(str, features)
    features = np.unique(features)
    # print('Final features set:\n')
    # for f in features:
    #     print("\t-{}".format(f))

    return features


def preprocess(data):

    data['time_remaining'] = 60 * data['minutes_remaining'] + data['seconds_remaining']

    data.sort_values(['game_date', 'period', 'time_remaining'], ascending = [True, True, False], inplace = True)

    data['dist'] = np.sqrt(data['loc_x'] ** 2 + data['loc_y'] ** 2)

    loc_x_zero = data['loc_x'] == 0
    data['angle'] = np.array([0] * len(data))
    data['angle'][~loc_x_zero] = np.arctan(data['loc_y'][~loc_x_zero] / data['loc_x'][~loc_x_zero])
    data['angle'][loc_x_zero] = np.pi / 2

    # match up - (away/home)
    data['home_play'] = data['matchup'].str.contains('vs').astype('int')

    # game date
    data['game_year'] = data['game_date'].dt.year
    data['game_month'] = data['game_date'].dt.month

    # Loc_x, and loc_y binning
    # data['loc_x'] = pd.cut(data['loc_x'], 25)
    # data['loc_y'] = pd.cut(data['loc_y'], 25)

    categorical_features = [
        'action_type',
        'combined_shot_type',
        'period',
        'season',
        'shot_type',
        # 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
        'game_year',
        'game_month',
        'opponent',
        # 'loc_x', 'loc_y'
    ]

    features = [data['time_remaining'], data['home_play'], data['dist'], data['angle']]
    for cc in categorical_features:
        dummies = pd.get_dummies(data[cc], prefix='{}#'.format(cc))
        features.append(dummies)

    new_data = pd.concat(features, axis=1)
    new_data.set_index(data.index.values, inplace=True)
    return new_data


data = pd.read_csv('data/data.csv', parse_dates=['game_date'])

data.set_index('shot_id', inplace=True)

# Replace 20 least common action types with value 'Other'
rare_action_types = data['action_type'].value_counts().sort_values().index.values[:20]
data.loc[data['action_type'].isin(rare_action_types), 'action_type'] = 'Other'


preprocessed_data = preprocess(data)
data = data.reindex(preprocessed_data.index.values)
unknown_mask = np.isnan(data['shot_made_flag'])

X_train = preprocessed_data[~unknown_mask]
y_train = data['shot_made_flag'][~unknown_mask]

X_test = preprocessed_data[unknown_mask]

final_features = select_features(X_train, y_train)

X_train = X_train.loc[:, final_features]
X_test = X_test.loc[:, final_features]

#{'max_depth': 8.0, 'min_samples_split': 5, 'max_features': 0.5, 'n_estimators': 100, 'criterion': 'gini', 'min_samples_leaf': 3}
rf = RandomForestClassifier(n_estimators=100,
                               criterion='gini',
                               min_samples_leaf=3,
                               min_samples_split=5,
                               max_features=0.5,
                               max_depth=8.0)

model = rf

# model = RandomForestClassifier(n_estimators=500, max_features=10)

# tree = DecisionTreeClassifier(min_samples_leaf=3,
#                                 min_samples_split=8, max_depth=5.0, max_features=0.8)
#
# model = BaggingClassifier(base_estimator=tree, n_estimators=500, random_state=1)

# model = RandomForestClassifier(n_estimators=500, criterion='entropy', min_samples_leaf=3,
#                                min_samples_split=8, max_depth=5.0, max_features=0.8)

# model = AdaBoostClassifier(n_estimators=500, random_state=0)

# model = DecisionTreeClassifier()

# model = RandomForestClassifier()

# model = svm.SVC()
# model.fit(X_train, y)

# param_grid = [{'n_estimators': [100, 150, 200, 300],
#                'criterion': ['gini', 'entropy'],
#                'min_samples_split': [3, 5, 8, 10],
#                'min_samples_leaf': [3, 5, 8, 10],
#                 'max_depth':[3.0, 5.0, 8.0, 10.0],
#                 'max_features':[0.3, 0.5, 0.8]
#                }]


# gs = GridSearchCV(estimator=RandomForestClassifier(),
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=2,
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# print(gs.best_params_)

cross_validation_report(X_train, y_train, model)

# create_submission(model, X_train, y_train, X_test, 'data/submission.csv')