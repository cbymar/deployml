def FitModel(X, Y, algorithm, gridSearchParams, cv):
    """Split, take a dict of gridsearch params,"""
    np.random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    grid = GridSearchCV(
        estimator=algorithm,
        param_grid=gridSearchParams,
        cv=cv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
    )

    grid_result=grid.fit(x_train, y_train)
    best_params=grid_result.best_params_
    pred = grid_result.predict(x_test)
    cm = confusion_matrix(y_test, pred)

    print(pred)
    pickle.dump(grid_result, open(algo_name, "wb"))

    print('Best Params :',best_params)
    print('Classification Report :',classification_report(y_test,pred))
    print('Accuracy Score : ' + str(accuracy_score(y_test,pred)))
    print('Confusion Matrix : \n', cm)

