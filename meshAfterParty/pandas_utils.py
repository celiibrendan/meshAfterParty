"""
Purpose: To make pandas table manipulation easier
"""

"""
Some useful functions: 
#sets the index to a specific column and doesn't return dataframe but modifies the existing one
df.set_index('Mountain', inplace=True) 

Selecting Data: 
i) selecting columns
- all columns are attribtues so can select with the "." (Ex: df.myCol)
- if there are wierd spaces could do getattr(df, 'Height (m)')
- BEST WAY: df['Height (m)'] because can select multiple columns at a time:
        df[['Height (m)', 'Range', 'Coordinates']]

ii) Selecting Rows (with slicing): 
- Can do that with index: a[start:stop:step]  (Ex: df[2:8:2])
- if you changed he index to be a certain column with non-numeric values: df['Lhotse':'Manaslu']
        slicing is inclusive at both start and end

iii) Specific data points: 
a) ** df.iloc** When just want to use numbers to reference
- df.iloc[rows, columns] for rows/col it can be: single int, list of ints, slices, : (for all)
  Ex: df.iloc[:, 2:6]** when 
  
b) ** df.loc where can use the names of things and not just numbers**
- df.loc[rows,columns]
   rows: index label, list of index labels, slice of index labels,:
   cols; singl column name, list of col names, slice of col names, :
   --> remember if do slice then it is inclusive on both start and stop
   
Ex: df.loc[:,'Height (m)':'First ascent']


c) Boolean selection: 
- can select rows with true/false mask--> 1) Create true false mask, put inside df[ ]
Ex: df[df['Height (m)'] > 8000]

For putting multiple conditions together the operators are: &,|,~

Have to use parenthesis to seperate multiple conditions: 
df[(df['Height (m)'] > 8000) & (df['Range']=='Mahalangur Himalaya')]

- Can use the loc operator to apply the mask and then select subset of columns: 
df.loc[(df['Height (m)'] > 8000) & (df['Range']=='Mahalangur Himalaya'), 'Height (m)':'Range']

Can also select the columns using a boolean operator: 
- col_criteria = [True, False, False, False, True, True, False]
  df.loc[df['Height (m)'] > 8000, col_criteria]
  
  
- To delete a column:
del neuron_df["random_value"]




------------------------------------pd.eval, df.eval df.query ------------------------------
Purpose: All of these functions are used to either
1) create masks of your rows
2) filter for specific rows 
3) make new data columns

pd.query:
- Can reference other dataframes by name and their columns with the attribute "."
pd.eval("df1.A + df2.A")   # Valid, returns a pd.Series object
pd.eval("abs(df1) ** .5")  # Valid, returns a pd.DataFrame object

- can evaluate conditional expressions: 
pd.eval("df1 > df2")        
pd.eval("df1 > 5")    
pd.eval("df1 < df2 and df3 < df4")      
pd.eval("df1 in [1, 2, 3]")
pd.eval("1 < 2 < 3")


Arguments that are specific:
- can use & or and (same thing)

** parser = "python" or "pandas" **
pandas: evaluates some Order of Operations differently
> is higher proirity than &
== is same thing as in  Ex: pd.eval("df1 in [1, 2, 3]") same as pd.eval("df1 == [1, 2, 3]")

python: if want traditional rules


** engine = "numexpr" or "python" ***

python: 
1) can do more inside your expressions but it is not as fast
Example: 
df = pd.DataFrame({'A': ['abc', 'def', 'abacus']})
pd.eval('df.A.str.contains("ab")', engine='python')

2) can reference variables with dictionary like syntax (not worth it)



---------- things can do with pd.eval -----------
1) Pass in dictionary to define variables that not defined 
        (otherwise uses global variable with same name)

pd.eval("df1 > thresh", local_dict={'thresh': 10})


---------- things can do with pd.eval -----------
1) only have to write column names in queries because only applied
to one dataframe so don't need to put name in front

df1.eval("A + B")

2) Have to put @ in front of variables to avoid confusion with column names
A = 5
df1.eval("A > @A") 


3) Can do multiline queries and assignments:

df1.eval('''
E = A + B
F = @df2.A + @df2.B
G = E >= F
''')

Can still do the local dict:
returned_df.query("n_faces_branch > @x",local_dict=dict(x=10000))

----------- Difference between df1.eval and df1.query -----------
if you are returning a True/False test, then df1.eval will just
return the True/False array whereas df1.query will go one step further
and restrict the dataframe to only those rows that are true

AKA: df1.eval is an intermediate step of df1.query 
(could use the df1.eval output to restrict rows by df1[output_eval])


---------------------------Examples--------------------------

Ex_1: How to check if inside a list
list_of_faces = [1038,5763,7063,11405]
returned_df.query("n_faces_branch in @list_of_faces",local_dict=dict(list_of_faces=list_of_faces))

Ex_2: Adding Ands/Ors

list_of_faces = [1038,5763,7063,11405]
branch_threshold = 31000
returned_df.query("n_faces_branch in @list_of_faces or skeleton_distance_branch > @branch_threshold",
                  local_dict=dict(list_of_faces=list_of_faces,branch_threshold=branch_threshold))
                  
Ex 2: of how to restrict a column to being a member or not being in a list
cell_df.query("not segment_id in @error_segments",local_dict=dict(error_seegments=error_segments))
"""

from pandas import util

def random_dataframe():
    return util.testing.makeDataFrame()

def random_dataframe_with_missing_data():
    util.testing.makeMissingDataframe()
    
def dataframe_to_row_dicts(df):
    return df.to_dict(orient='records')

def n_nans_per_column(df):
    return df.isnull().sum()

def n_nans_total(df):
    return df.isnull().sum().sum()

def surpress_scientific_notation():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
import pandas as pd
import pandasql
def restrict_pandas(df,index_restriction=[],column_restriction=[],value_restriction=""):
    """
    Pseudocode:
    How to specify:
    1) restrict rows by value (being equal,less than or greater) --> just parse the string, SO CAN INCLUDE AND
    2) Columns you want to keep
    3) Indexes you want to keep:
    
    Example: 
    
    index_restriction = ["n=50","p=0.25","erdos_renyi_random_location_n=100_p=0.10"]
    column_restriction = ["size_maximum_clique","n_triangles"]
    value_restriction = "transitivity > 0.3 AND size_maximum_clique < 9 "
    returned_df = restrict_pandas(df,index_restriction,column_restriction,value_restriction)
    returned_df
    
    Example of another restriction = "(transitivity > 0.1 AND n_maximal_cliques > 10) OR (min_weighted_vertex_cover_len = 18 AND size_maximum_clique = 2)"
    
    
    #how to restrict by certain value in a column being a list 
    df[~df['stn'].isin(remove_list)]
    """
    new_df = df.copy()
    
    
    if len(index_restriction) > 0:
        list_of_indexes = list(new_df[graph_name])
        restricted_rows = [k for k in list_of_indexes if len([j for j in index_restriction if j in k]) > 0]
        #print("restricted_rows = " + str(restricted_rows))
        new_df = new_df.loc[new_df[graph_name].isin(restricted_rows)]
        
    #do the sql string from function:
    if len(value_restriction)>0:
        s = ("SELECT * "
            "FROM new_df WHERE "
            + value_restriction + ";")
        
        #print("s = " + str(s))
        new_df = pandasql.sqldf(s, locals())
        
    #print(new_df)
    
    #restrict by the columns:
    if len(column_restriction) > 0:
        #column_restriction.insert(0,graph_name)
        new_df = new_df[column_restriction]
    
    return new_df

def turn_off_scientific_notation(n_decimal_places=3):
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    
def find_all_rows_with_nan(df,return_indexes=True):
    if return_indexes:
        return np.where(df.isna().any(axis=1))[0]
    else:
        return df[df.isna().any(axis=1)]
    
def filter_away_nan_rows(df):
    return df[~(df.isna().any(axis=1))]
    
from IPython.display import display
def display_df(df):
    display(df)
    
def dicts_to_dataframe(list_of_dicts):
    return pd.DataFrame.from_dict(list_of_dicts)
    
# ----------- 1/25 Additon: Used for helping with clustering ----------- #
import numpy_utils as nu
import matplotlib.pyplot as plt
import numpy as np

def divide_dataframe_by_column_value(df,
                                column,
                                ):
    """
    Purpose: To divide up the dataframe into 
    multiple dataframes
    
    Ex: 
    divide_dataframe_by_column_value(non_error_cell,
                                column="cell_type_predicted")
    
    """
    tables = []
    table_names = []
    for b,x in df.groupby(column):
        table_names.append(b)
        tables.append(x)
        
    return tables,table_names


def plot_histogram_of_differnt_tables_overlayed(tables_to_plot,
                          tables_labels,
                          columns=None,
                          fig_title=None,
                           fig_width=18.5,
                            fig_height = 10.5,
                            n_plots_per_row = 4,
                            n_bins=50,
                            alpha=0.4,
                                           density=False):
    """
    Purpose: Will take multiple
    tables that all have the same stats and to 
    overlay plot them
    
    Ex: plot_histogram_of_differnt_tables_overlayed(non_error_cell,"total",columns=stats_proj)
    
    """
    if not nu.is_array_like(tables_to_plot):
        tables_to_plot = [tables_to_plot]
        
    if not nu.is_array_like(tables_labels):
        tables_labels = [tables_labels]
    
    ex_table = tables_to_plot[0]
    
    if columns is None:
        columns = list(cell_df.columns)
        
    n_rows = int(np.ceil(len(columns)/n_plots_per_row))
    
    fig,axes = plt.subplots(n_rows,n_plots_per_row)
    fig.set_size_inches(fig_width, fig_height)
    fig.tight_layout()
    
    if not fig_title is None:
        fig.title(fig_title)

    
    for j,col_title in enumerate(columns):
        
        row = np.floor(j/4).astype("int")
        column = j - row*4
        ax = axes[row,column]
        ax.set_title(col_title)
        
        for curr_table,curr_table_name in zip(tables_to_plot,tables_labels):
            curr_data = curr_table[col_title].to_numpy()
            ax.hist(curr_data,bins=n_bins,label=curr_table_name,alpha=alpha,
                    density=density)
            
        ax.legend()
        
def plot_histograms_by_grouping(df,
                               column_for_grouping,
                               **kwargs):
    
    dfs,df_names = divide_dataframe_by_column_value(df,
                                column=column_for_grouping,
                                                   )
    
    plot_histogram_of_differnt_tables_overlayed(tables_to_plot=dfs,
                                               tables_labels=df_names,
                                               **kwargs)
    
