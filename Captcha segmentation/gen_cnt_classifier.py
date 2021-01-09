'''
建立SVM模型，用于预测连通域中所包含的字符数量，将模型直接保存入文件中，在contours.py中直接调用
'''

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, GridSearchCV
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    import pickle

    # Load the data
    try:
        data = pd.read_csv('./contours-data.csv')
    except:
        raise Exception('Failed to load contours dataset')

    # Get features and outcome variable
    X = data[['bbox_width', 'bbox_height', 'area', 'extent', 'perimeter']]
    y = data['num_chars'].values.astype(np.uint8)

    # Normalize features
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)

    # Split data into train & test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=0.2)

    # Create the estimator
    gds = GridSearchCV(estimator=SVC(gamma='auto', probability=True),
                param_grid={
                    'C': np.linspace(1, 10, 20)
                },
                n_jobs=-1,
                cv=RepeatedStratifiedKFold(5, 3),
                scoring='accuracy')

    gds.fit(X_train, y_train)

    model = gds.best_estimator_


    # Evaluate the model on train / test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Show classification reports
    print('Results over train set:')
    print(classification_report(y_train, y_train_pred))

    print('Results over test set:')
    print(classification_report(y_test, y_test_pred))

    # Save the model to a file
    with open('.contour-classifier', 'wb') as file:
        pickle.dump(model, file)

    with open('.contour-classifier-preprocessor', 'wb') as file:
        pickle.dump(std_scaler, file)
