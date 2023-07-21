import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['Pack Price', 'Weight (Kilograms)']
            categorical_columns = ['Country', 'Fulfill Via', 'Vendor INCO Term','Vendor', 'Shipment Mode','Sub Classification', 'First Line Designation']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent", fill_value='missing')),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            # target_column_name="Freight Cost (USD)"
            # numerical_columns = ['Pack Price', 'Weight (Kilograms)']

            # input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            # target_feature_train_df=train_df[target_column_name]

            # input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            # target_feature_test_df=test_df[target_column_name]

            
            logging.info("Applying preprocessing object on training dataframe")
            train_arr = preprocessing_obj.fit_transform(train_df)

            logging.info("Applying preprocessing object on testing dataframe")
            test_arr = preprocessing_obj.transform(test_df)

            # logging.info(f"input_feature_train_arr shape: {input_feature_train_arr.shape}")
            # logging.info(f"input_feature_test_arr shape: {input_feature_test_arr.shape}")
            
            # target_feature_train_arr = np.array(target_feature_train_df)
            # target_feature_test_arr = np.array(target_feature_test_df)
            
            # logging.info(f"target_feature_train_arr shape: {target_feature_train_arr.shape}")
            # logging.info(f"target_feature_test_arr shape: {target_feature_test_arr.shape}")

            # target_feature_train_arr = target_feature_train_arr.reshape(-1, 1)
            # target_feature_test_arr = target_feature_test_arr.reshape(-1, 1)
            # print(input_feature_train_arr.dtype)
            # print(input_feature_test_arr.dtype)
            # print(target_feature_train_arr.dtype)
            # print(target_feature_test_arr.dtype)

            # train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr[:, np.newaxis]), axis=1)
            # test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr[:, np.newaxis]), axis=1)

            # train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info(f"train_arr shape: {train_arr.shape}")
            logging.info(f"test_arr shape: {test_arr.shape}")

            # train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            # test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)