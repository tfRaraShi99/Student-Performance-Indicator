import sys
import pandas as pd 
import numpy as np 
import os 

from src.utils import load_object
from src.exception import CustomException 
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
          try:
            model_path = os.path.join('artifact',"model.pkl")
            preprocessor_path = os.path.join('artifact','preprocessor.pkl')

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Log the features before preprocessing
            logging.info(f"Input features before handling missing values:\n{features.head()}")

            # Impute missing values in categorical columns
            for col in features.select_dtypes(include=['object']).columns:
                  if features[col].isnull().any():
                        logging.info(f"Imputing missing values in column: {col}")

                        # Check if mode exists; if not, use a fallback value
                        if not features[col].mode().empty: 
                            features[col].fillna(features[col].mode()[0], inplace=True)
                        else:
                            # Fallback value for empty columns
                            features[col]=features[col].fillna("Unknown")  

            # Log the features after handling missing values
            logging.info(f"Input features after handling missing values:\n{features.head()}")

            data_scaled = preprocessor.transform(features)
            logging.info("Data transformation completed successfully.")
            
            pred=model.predict(data_scaled)
            logging.info("Prediction completed successfully.")

            return pred
          except Exception as e:
                raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):
                
                self.gender = gender
                self.race_ethnicity = race_ethnicity
                self.parental_level_of_education = parental_level_of_education
                self.lunch = lunch
                self.test_preparation_course = test_preparation_course
                self.reading_score = reading_score
                self.writing_score = writing_score


    def get_data_as_dataframe(self):
          try:
            custom_data_input_dict = {
                'gender' : [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education' : [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
                }
            # Create DataFrame and log the input data
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Custom data converted to DataFrame:\n{df}")

            return df
          
          except Exception as e:
                raise CustomException(e,sys)
        