
# --------------------  The whole pandas to plots ---------------------- #
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
    
    """
    new_df = df.copy()
    
    
    #How to restrict by a list in "restricted_rows" (instead of one value)
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
