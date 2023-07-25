import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Country: str,
        Fulfill_Via: str,
        Vendor_INCO_Term: str,
        Shipment_Mode: str,
        Sub_Classification: str,
        Vendor: str,
        First_Line_Designation: str,
        Pack_Price: int,
        Year: str,
        Weight: int):

        self.Country = Country

        self.Fulfill_Via = Fulfill_Via

        self.Vendor_INCO_Term = Vendor_INCO_Term

        self.Shipment_Mode = Shipment_Mode

        self.Sub_Classification = Sub_Classification

        self.Vendor = Vendor

        self.First_Line_Designation = First_Line_Designation

        self.Pack_Price = Pack_Price
        
        self.Year = Year

        self.Weight = Weight

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Country": [self.Country],
                "Fulfill Via": [self.Fulfill_Via],
                "Vendor INCO Term": [self.Vendor_INCO_Term],
                "Shipment Mode": [self.Shipment_Mode],
                "Sub Classification": [self.Sub_Classification],
                "Vendor": [self.Vendor],
                "First Line Designation": [self.First_Line_Designation],
                "Pack Price": [self.Pack_Price],
                "Year":[self.Year],
                "Weight (Kilograms)": [self.Weight],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)