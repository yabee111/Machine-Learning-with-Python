# Machine-Learning-with-Python
TU/e Machine Learning Course Assignment Project

[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)]
  

## Assignment Overview
### Assignment1: Logistic Regression & Cross-validation
Leveraging the power of Machine Learning, this assignment embarked on a journey to decipher ancient script characters from the [Kuzushiji dataset](https://www.openml.org/d/41982). Our mission involves not only recognizing these characters but also uncovering their intricate patterns. The ultimate goal is to translate them into modern Japanese (Hiragana) characters, breathing new life into a historical linguistic art form.


### Assignment2: Data Cleaning, Encoding, Pipeline & Feature Importance
Rather than merely conducting a grid search across an array of models, the assignment focus lies in deliberate and considerate preprocessing, using the [Employee Salary dataset](https://www.openml.org/d/42125), a comprehensive collection of salary information for individuals employed within a local government in USA. The initial step entails constructing a machine learning pipeline aimed at performing essential data preprocessing. This process ensures that we can meticulously analyze models in a purposeful manner, all the while safeguarding against data leakage during the evaluation phase. The ultimate objective is to predict salaries and, simultaneously, scrutinize the data and our models for any potential biases. This scrutiny aids us in developing an awareness of potential biases and equips us with the knowledge to circumvent them during model training.

#### Example Code: Pipeline
```
def flexible_pipeline(X, model, scaler=StandardScaler(), encoder=OneHotEncoder(sparse=False, handle_unknown='ignore')):
    # Numeric Features
    numerical = X.select_dtypes(exclude=["category","object"]).columns.tolist()
    # Categorical Features
    categorical = X.select_dtypes(include=["category", "object"]).columns.tolist()
    
    if scaler == None:
        numerical_pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value = 0.00))
    else:
        numerical_pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value = 0.00), scaler)      
    categorical_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"), encoder)
    transformer = make_column_transformer((numerical_pipe, numerical), (categorical_pipe, categorical))
    
    return Pipeline(steps=[("preprocessing", transformer), ("model", model)])
```

#### Example Code: Random Forest and Feature Importance 
```
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True)

    X_sub = X_train.copy()
    y_sub = y_train.copy()
    
    # Combine flexible_pipeline with one-hot-encoding, with a RandomForest regressor. Train that pipeline on the training set.
    clf = flexible_pipeline(X_sub, model = RandomForestRegressor(), encoder = OneHotEncoder(sparse=False, handle_unknown='ignore'), scaler = None).fit(X_sub, y_sub)
    # input_features=categorical
    categorical = X.select_dtypes(include=["category", "object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["category","object"]).columns.tolist()

    # Remember that the categorical features were encoded. 
    rf_feature_names = clf.named_steps.preprocessing.named_transformers_['pipeline-2'].named_steps['onehotencoder'].get_feature_names()
    rf_feature_names = np.insert(rf_feature_names, 0, numerical)

    # Retrieve the feature importances from the trained random forest and match them to the correct names. 
    rf_feature_importances = clf.named_steps.model.feature_importances_ 
    
    # Compute the permutation importances given the random forest pipeline and the test set. Use random_state=0 and at least 10 iterations.
    permutation_importances = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state=0)
```

### Assignment3: Convolutional Neural Network, Image Preprocessing & Recognition, Tsne & Mobilenetv2
In this assignment, we'll harness the capabilities of a TensorFlow Dataset to drive our exploration. Specifically, we've opted to engage with the [rock_paper_scissors dataset](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors), which boasts a collection of captivating images depicting hands in the midst of rock, paper, and scissor gameplay. Each of these images boasts dimensions of (300, 300, 3), and the dataset itself encompasses a robust assemblage of 2520 training images and 372 testing images.

#### Example Code: Mobilenetv2
```
def build_model_3_1():
    conv_base = MobileNetV2(input_shape = IMG_SHAPE, include_top=False)

    conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, kernel_regularizer='l2', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
```



##### Reference: [Tu/e 2AMM15 Machine Learning Course](https://github.com/ML-course)
