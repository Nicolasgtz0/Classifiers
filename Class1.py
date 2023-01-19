import pandas

def clean_data(path: str) -> pd.DataFrame: 
    """Clean data and return only the neccesary columns
    Args:
     path (str): location of the file on our computers 
    Returns:
    pd.DataFrame: the output dataframe with the correct columns 
    """

    assert path[-4:] == '.csv'
    df = pd.read_csv(path)
    