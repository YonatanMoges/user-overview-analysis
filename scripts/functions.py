import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from IPython.display import Image
""" import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go   """


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
  




def hist(df:pd.DataFrame, column:str, color:str)->None:
    plt.figure(figsize=(9, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()


def box_plot(df: pd.DataFrame, x_col: str, title: str)->None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=90, fontsize=14)
    plt.show()


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 10000)
    plt.ylim(0, 10000)
    plt.show()


def plot_heatmap(df: pd.DataFrame, title: str, cbar=False) -> None:
    plt.figure(figsize=(15, 12))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
    plt.title(title, size=18, fontweight='bold')
    plt.show()
    

def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].mode(),df[column])
    
    return df[column]


def plot_bar(column, title, xlabel, ylabel):
    plt.figure(figsize=(10,5))
    sns.barplot(x=column.index, y=column.values) 
    plt.title(title, size=14, fontweight="bold")
    plt.xlabel(xlabel, size=13, fontweight="bold") 
    plt.ylabel(ylabel, size=13, fontweight="bold")
    plt.xticks(rotation=90)
    plt.show() 


def mult_hist(sr, rows, cols, title_text, subplot_titles):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.suptitle(title_text, fontsize=16)

    # Flatten axes array in case rows or cols > 1
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i in range(rows * cols):
        x = ["-> " + str(i) for i in sr[i].index]
        axes[i].bar(x, sr[i].values)
        axes[i].set_title(subplot_titles[i], fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Show the plot
    plt.show()


def plot_hist(df: pd.DataFrame, column: str, color: str) -> None:
    plt.figure(figsize=(9, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()


def scatter2d(df, x, y, c=None, s=None, mx=None, my=None, af=None, fit=None, interactive=False):
    fig = px.scatter(df, x=x, y=y, color=c, size=s, marginal_y=my,
                     marginal_x=mx, trendline=fit, animation_frame=af)
    if(interactive):
        st.plotly_chart(fig)
    else:
        st.image(pio.to_image(fig, format='png', width=1200))


def scatter3D(df, x, y, z, c=None, s=None, mx=None, my=None, af=None, fit=None, rotation=[1, 1, 1], interactive=False):
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=c, size=s,
                        animation_frame=af, size_max=18)

    fig.update_layout(scene=dict(camera=dict(eye=dict(x=rotation[0], y=rotation[1], z=rotation[2]))),
                      )
    if(interactive):
        st.plotly_chart(fig)
    else:
        st.image(pio.to_image(fig, format='png', width=1200))

def plot_elbow_method(distortions, inertias):
    # Create a figure with 1 row and 2 columns for subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the distortions on the first subplot
    axes[0].plot(range(1, 15), distortions, marker='o', linestyle='-', color='b')
    axes[0].set_title("Distortion", fontsize=14)
    axes[0].set_xlabel('Number of Clusters', fontsize=12)
    axes[0].set_ylabel('Distortion', fontsize=12)

    # Plot the inertias on the second subplot
    axes[1].plot(range(1, 15), inertias, marker='o', linestyle='-', color='r')
    axes[1].set_title("Inertia", fontsize=14)
    axes[1].set_xlabel('Number of Clusters', fontsize=12)
    axes[1].set_ylabel('Inertia', fontsize=12)

    # Add the overall title for the figure
    fig.suptitle("The Elbow Method", fontsize=16)

    # Display the plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_scatter(user_engagement, x_col, y_col, color_col, size_col, marker='o'):
    # Define the size of the figure
    plt.figure(figsize=(12, 8))

    # Scatter plot with different colors for clusters and sizes for session frequency
    scatter = plt.scatter(
        x=user_engagement['Total_Data_Volume'],
        y=user_engagement['Duration'],
        c=user_engagement['Cluster'].astype('category').cat.codes,  # Convert clusters to numeric codes for coloring
        s=user_engagement['Session_Frequency'] * 20,  # Scale size for better visibility
        cmap='viridis',  # Colormap for clusters
        alpha=0.7  # Slight transparency for better visibility
    )

    # Add colorbar to indicate cluster codes
    plt.colorbar(scatter, label='Cluster')

    # Set labels and title
    plt.xlabel('Total Data Volume')
    plt.ylabel('Duration')
    plt.title('User Engagement Scatter Plot')

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_scatter(df, x_col, y_col, color_col, size_col):
    # Define the size of the figure
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot
    scatter = plt.scatter(
        x=df[x_col],
        y=df[y_col],
        c=df[color_col].astype('category').cat.codes,  # Convert color column to numeric codes
        s=df[size_col] * 20,  # Scale size for better visibility
        cmap='viridis',  # Colormap for clusters
        alpha=0.7,  # Slight transparency for better visibility
        marker=marker  # Directly apply the marker style
    )
    
    # Add colorbar to indicate cluster codes
    plt.colorbar(scatter, label=color_col)
    
    # Set labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('User Engagement Scatter Plot')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_scatter_pandas(df, x_col, y_col, color_col, size_col):
    # Create a figure and axis
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    scatter = plt.scatter(
        x=df[x_col],
        y=df[y_col],
        c=df[color_col].astype('category').cat.codes,  # Convert color column to numeric codes
        s=df[size_col] * 2,  # Adjust size for visibility
        cmap='viridis',  # Use the same colormap as in plotly
        alpha=0.7  # Transparency
    )
    
    # Add colorbar to show clusters
    plt.colorbar(scatter, label=color_col)
    
    # Set labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('User Engagement Scatter Plot')

    # Tight layout
    plt.tight_layout()
    
    # Save the plot as an image
    #plt.savefig('scatter_plot.png', format='png', dpi=300)
    
    # Show the plot
    plt.show()