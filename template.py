#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data Science Project Skeleton

This is a simple template to provide a general structure for Data Science projects.
It outlines the common phases of a data science project from prototyping to production
and prescribes boundaries that may easily be followed by data scientists and machine
learning engineers.  

It is recommended that a project should contain these top-level functions, such that
each function can serve as an entry point for running a process or batch job
by a separate driver script/program. Module imports are recommended to be defined inside
each of the top-level functions. This does not mean that practitioners shouldn't practice
modular coding and development best practices.

Example driver program:

>> # Discovery or Model Prototyping Phase
>> raw_data = load_data(from='2018-01-01', until='2020-01-01')
>> train, test = preprocess_data(raw_data, train_size=0.8)
>> train_augmented, feature_generator = create_features(train, is_train=True)
>> test_augmented = create_features(test, is_train=False,
                                        feature_generator=feature_generator)
>> model = generate_model(train_augmented)
>> preds = generate_predictions(model=model, data=test_augmented)
>> reports = produce_reports(test, preds)

>> # Finalizing Model Training for Generating Predictions
>> raw_data = load_data(from='2018-01-01', until='2020-02-01')
>> train, _test = preprocess_data(raw_data, train_size=1)
>> train_augmented, feature_generator = create_features(train, is_train=True)
>> model = generate_model(train_augmented)
>> incoming_data = load_data(from='2018-02-01', until='2020-02-06')
>> incoming_data_augmented = create_features(incoming_data, is_train=False,
                                          feature_generator=feature_generator)
>> preds = generate_predictions(model=model, data=incoming_data_augmented)
"""


def load_data(**args):
    """
    Load datasets from one or multiple sources of data, such as from the file system,
    Apache Spark, HDFS, etc.

    Args:
        **args: This should be replaced with useful and descriptive input arguments,
        such as file paths, URIs, credential paths, etc.

    Returns:
        This should return a serialized dataset, DataFrame, or absolute path of the
        dataset if persisted in another location.
    """
    pass


def preprocess_data(**args):
    """
    Preprocess the provided dataset to follow the desired format.
    This is where it is recommended to remove irrelevant features, such as IDs and
    protected/sensitive information, and filtering of the provided dataset.
    Data aggregation is also recommended in this phase but may also be moved into
    the create_features() phase, if unavoidable.

    Finally, this function is recommended to generate separate training and testing sets.

    Args:
        **args: This should be replaced with useful and descriptive input arguments,
        such as the location of the dataset to be preprocessed, as well as other input
        parameters.

    Returns:
        This should return serialized datasets, DataFrames, or absolute paths of the
        train and test datasets if persisted in another location.
    """
    pass


def create_features(**args):
    """
    Create or engineer derived features for the input dataset, such as imputing missing
    values, scaling features, generating embeddings, etc. It is 
    recommended to persist feature engineering objects in this phase, as well as to
    aid in tracking of these persisted objects in the user's chosen format. 

    This function is recommended to have an input argument to whether it should process a
    training or testing set, and return the augmented dataset. If the input is a
    training set, it is recommended to also return path/s or object/s of the stateful
    feature generators.

    Args:
        **args: This should be replaced with useful and descriptive input arguments,
        such as the location of the dataset to generate features, an input
        flag if the provided dataset is a training or testing set, and the path or
        object of the stateful feature generator.

    Returns:
        This should return the serialized dataset, DataFrame, or absolute path of the
        augmented dataset if persisted in another location. It is also recommended to
        return the object/s or paths/s of the feature generator.
    """
    pass


def generate_model(**args):
    """
    Train a model with the provided data, where cross-validation should ideally be
    implemented. Ideally, feature selection and reduction should be included in this phase.

    Args:
        **args: This should be replaced with useful and descriptive input arguments,
        such as the location of the dataset to train the model on, and other arguments
        specific to the model training phase.

    Returns:
        This should return the serialized model or its absolute path if persisted in
        another location. It is also recommended to return the object/s or paths/s
        of the serialized model such that is easy to import or use in multiple 
        independent contexts. Possibly look into the ONNX format or MLFlow Model
        Registry.
    """


def generate_prediction(**args):
    """
    Generate predictions from a provided model and dataset.

    Args:
        **args: This should be replaced with useful and descriptive input arguments,
        such as the location of the model and the dataset for the model to generate its
        predictions from, and other arguments specific to the model prediction.

    Returns:
        This should return the serialized predictions or its absolute path if persisted in
        another location. It is also recommended to return the object/s or paths/s
        of the serialized object that is easy to import or use in multiple independent
        contexts, like a CSV or JSON format.
    """
    pass


def produce_reports(**args):
    """
    Generate model performance reports, such as Precision, Recall, RMSE, LogLoss, etc.

    Args:
        **args: This should be replaced with useful and descriptive input arguments,
        such as the location of the reference dataset, the generated predictions of the
        model, and other arguments specific to the model prediction.

    Returns:
        This should return the serialized report or its absolute path if persisted in
        another location. It is also recommended to return the object/s or paths/s
        of the report so that it is easy to import or use in multiple independent
        contexts.
    """
    pass
