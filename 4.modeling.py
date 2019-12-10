# %% [markdown]
# **********************Modeling**********************
# %% [code]
#try a simple linear regression first
metric = 'neg_mean_squared_error'
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
print(f"{np.sqrt(-cross_val_score(LinearRegression(), train, y_train, cv=kfold, scoring=metric)).mean():.4f} Error")

# %% [code]
#Grid search of elastic net, random forest and xgb
#LightGBM takes forever to run
#KernelRidge generates errors: Mean Squared Logarithmic Error cannot be used when targets contain negative values
#ElasticNet
elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0004, l1_ratio=1, random_state=1))
param_grid = {
    'elasticnet__alpha' : np.linspace(0.0001, 0.001, 10),
    'elasticnet__l1_ratio' : np.linspace(0.2, 2, 20),
}
search = GridSearchCV(elastic, param_grid, cv=10, scoring=metric, n_jobs=-1)
search.fit(train, y_train)
best_params_ela = search.best_params_
print(f"{search.best_params_}")
print(f"{np.sqrt(-search.best_score_):.4}")

# %% [code]
#random forest
rdf = make_pipeline(RobustScaler(), RandomForestRegressor(max_depth=4, n_estimators=100, random_state=1))
param_grid={
            'max_depth': range(2,4),
            'n_estimators': (50, 100),
        }
search = GridSearchCV(RandomForestRegressor(), param_grid, cv=10, scoring=metric, n_jobs=-1)   
search.fit(train, y_train)
best_params_forest = search.best_params_
print(f"{search.best_params_}")
print(f"{np.sqrt(-search.best_score_):.4}")

# %% [code]
#XGBboost
xgbreg = xgb.XGBRegressor(objective="reg:squarederror",
                             colsample_bytree=0.45, gamma=0.046, 
                             learning_rate=0.03, max_depth=2, 
                             min_child_weight=0.4, n_estimators=10,
                             reg_alpha=0.47, reg_lambda=0.8,
                             subsample=0.5, random_state=1, n_jobs=-1)

param_grid = {
    'xgb__max_depth' : [2, 3],
    'xgb__estimators' : [10, 25, 50],
    "xgb__learning_rate" : [0.01, 0.02],
    "xgb__min_child_weight" : [0.2, 0.3],
    }
search = GridSearchCV(xgbreg, param_grid, cv=3, scoring=metric, n_jobs=-1)
search.fit(train, y_train)
best_params_xgb = search.best_params_
print(f"{search.best_params_}")
print(f"{np.sqrt(-search.best_score_):.4}")

# %% [code]
elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=best_params_ela['elasticnet__alpha'], 
                                                   l1_ratio=best_params_ela['elasticnet__l1_ratio'], random_state=1))
forest = make_pipeline(RobustScaler(), RandomForestRegressor(max_depth=best_params_forest["max_depth"], n_estimators=best_params_forest["n_estimators"],  random_state=1))
xgb_reg = xgb.XGBRegressor(objective="reg:squarederror",
                             colsample_bytree=0.45, gamma=0.046, 
                             max_depth=best_params_xgb["xgb__max_depth"], n_estimators=best_params_xgb["xgb__estimators"], 
                             learning_rate = best_params_xgb['xgb__learning_rate'], min_child_weight= best_params_xgb["xgb__min_child_weight"],
                             reg_alpha=0.47, reg_lambda=0.8,
                             subsample=0.5, random_state=1, n_jobs=-1)

# %% [code]
#create a meta model
classifiers = [elastic_net, forest, xgb_reg]
clf_names   = ["elastic_net", "random_forest", "XGBboost"]
weights = [0.8, 0.15, 0.05]

predictions_exp= []
for clf_name, clf, weight in zip(clf_names, classifiers, weights):
    print(f"{clf_name} {np.sqrt(-cross_val_score(clf, train, y_train, scoring=metric).mean()):.5f}")
    clf.fit(train, y_train)
    preds = clf.predict(test)
    predictions_exp.append(weight*np.expm1(preds))
prediction_final = pd.DataFrame(predictions_exp).sum().T.values

# %% [code]
#submit to see the final MSE
submission = pd.DataFrame({'Id': test_ID, 'SalePrice': prediction_final})
submission.to_csv("submission.csv", index=False)
