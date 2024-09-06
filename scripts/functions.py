import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_csv(file_name):
  try:
    null_values = ["n/a", "na", "undefined"]
    df = pd.read_csv(f'../data/{file_name}', na_values=null_values)
    return df
  except:
    print("Log:-> File Not Found")

def save_file(file_name):
  pass


class Marketing():
  def __init__(self, df) -> None:
      self.df = df.copy()

  
  def filter_necessary_columns(self):
    '''
    Make data frame to only contain necessary columns for marketing
    '''
    columns = ['MSISDN/Number', 'Handset Type', 'Handset Manufacturer']
    self.df = self.df[columns]

  
  def get_top_manufacturers(self, top=3):
    top_manufacturers = self.df.groupby("Handset Manufacturer").agg({"MSISDN/Number":'count'}).reset_index()
    top_manufacturers = top_manufacturers.sort_values(by='MSISDN/Number', ascending=False).head(top)
    return top_manufacturers

  
  def get_top_handsets(self, top=10):
    top_handset = self.df.groupby("Handset Type").agg({"MSISDN/Number":'count'}).reset_index()
    top_handset = top_handset.sort_values(by='MSISDN/Number', ascending=False).head(top)
    return top_handset


  def get_best_phones(self):
    top_3_manufacturers = self.get_top_manufacturers(3)

    manufacturers = self.df.groupby("Handset Manufacturer")

    for column in top_3_manufacturers['Handset Manufacturer']:
      result = manufacturers.get_group(column).groupby("Handset Type")['MSISDN/Number'].nunique().nlargest(5)
      print(f"**** { column } ***")
      print(result)
      print()




def plot_hist(df:pd.DataFrame, column:str, color:str, ax)->None:
    # plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.displot(data=df, x=column, color=color, kde=True, ax=ax)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def hist(df:pd.DataFrame, column:str, color:str)->None:
    plt.figure(figsize=(9, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()
    

def plot_count(df:pd.DataFrame, column:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x=column)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()
    
def plot_bar(df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str, ax)->None:
    plt.figure(figsize=(12, 7))
    sns.barplot(data = df, x=x_col, y=y_col, ax=ax)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()

def plot_heatmap(df:pd.DataFrame, title:str, cbar=False)->None:
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
    plt.title(title, size=18, fontweight='bold')
    plt.show()

def plot_box(df:pd.DataFrame, x_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.show()

def plot_box_multi(df:pd.DataFrame, x_col:str, y_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()

def serious_bar(serious, ax):
    '''
    Plot bar chart for serious data types
    '''
    
    return sns.barplot(x=serious.index, y=serious, ax=ax)
  

