# importing library
import pandas as pd
import numpy as np
# creating class
class DataFrameCleaning():
    def __init__(self, df):
        self.df = df.copy()
        print('Automation in Action...Great!!!')
    

    def get_column_with_many_null(self):
        '''
        Return List of Columns which contain more than 30% of null values
        '''
        df_size = self.df.shape[0]
        
        columns_list = self.df.columns
        many_null_columns = []
        
        for column in columns_list:
            null_per_column = self.df[column].isnull().sum()
            percentage = round((null_per_column / df_size) * 100 , 2)
            
            if(percentage > 30):
                many_null_columns.append(column)
        
        return many_null_columns
    

    def drop_columns(self, columns):
        '''
        Return Dataframe with Most null columns removed.
        '''

        self.df.drop(columns, axis=1, inplace=True)


    def convert_to_datetime(self, df):
        """
        convert start and end column to datetime
        """

        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])

        return df


    def drop_duplicate(self, df):
        """
        drop duplicate rows
        """
        df.drop_duplicates(inplace=True)

        return df


    def drop_rows(self, columns):
        '''
        Drop Rows of specified columns, which contain null values
        '''
        self.df.dropna(subset=columns, inplace=True)

    
    def fill_numerical_column(self, column):
        '''
        Fill Numerical null values with mean or median based on the skewness of the column
        '''
 
        for col in column:
            skewness = self.df[col].skew() 
            if((-1 < skewness) and (skewness < -0.5)):
                self.df[col] = self.df[col].fillna(self.df[col].mean()) 

            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())

    
    def fill_categorical_columns(self, column):
        '''
        Fill Categorical null values with column Mode
        '''
 
        for col in column:
            mode = self.df[col].mode()[0]
            self.df[col] = self.df[col].fillna(mode)


class DataFrameInfo():
    def __init__(self, df):
        self.df = df.copy()


    def get_columns_list(self):
        '''
        Return Column list of the Dataframe
        '''
        return self.df.columns.to_list()


    def detail_info(self):
        '''
        Display the detail of the DataFrame information
        '''

        print(self.df.info())


    def null_column_percentage(self):
        '''
        Display Total Null percentage of the Data Frame Columns
        '''

        num_rows, num_columns = self.df.shape
        df_size = num_rows * num_columns
        
        null_size = (self.df.isnull().sum()).sum()
        percentage = round((null_size / df_size) * 100, 2)
        print(f"The Telecom data contains { percentage }% missing values.")


    def get_null_counts(self):
        '''
        Display Null Counts of each column
        '''

        print(self.df.isnull().sum())

#df.skew().sort_values(ascending=False)
    def skewness(self):
        '''
        Display The skew value of each columns 
        '''
        print(self.df.skew())

    
def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].mode(),df[column])

    return df[column]


def replace_outliers_with_iqr(df, columns):
    for col in columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        cut_off = IQR * 1.5
        lower, upper = Q1 - cut_off, Q3 + cut_off

        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])

import pandas as pd

class UserOverviewScript():
    def __init__(self, df) -> None:
        self.df = df.copy()


    def get_top_handsets(self,  num):
        top_handset = self.df['Handset Type'].value_counts().head(num)
        return top_handset


    def get_top_manufacturers(self,  num):
        top_handset = self.df['Handset Manufacturer'].value_counts().head(num)
        return top_handset

    
    def get_handset_group(self):
        top_3_manufacturers = self.get_top_manufacturers(3)

        manufacturers = self.df.groupby("Handset Manufacturer")

        for column in top_3_manufacturers.index:
            result = manufacturers.get_group(column).groupby("Handset Type")['MSISDN/Number'].nunique().nlargest(5)
            print(f">>>> { column } <<<<")
            print(result)
            print() 

    def convert_bytes_to_megabytes(self, column):
        """
            This function takes the dataframe and the column which has the bytes values
            returns the megabytesof that value            
        """        
        megabyte = 1*10e+5
        Total_MB = []
        for i in column.values:
            i = i / megabyte
            Total_MB.append(i)

        return Total_MB


    def convert_bytes_to_kbytes(self, column):
        """
            This function takes the dataframe and the column which has the bytes values
            returns the kilobytes of that value            
        """        
        Total_kb = []
        for i in column.values:
            i = i / 1024
            Total_kb.append(i)

        return Total_kb


    def convert_ms_to_sec(self, column):
        """
            This function takes the dataframe and the millisecond column values
            returns the second equivalence          
        """        
        
        Total_sec = []
        for i in column.values:
            i = (i / 1000) % 60
            Total_sec.append(i)

        return Total_sec
    
