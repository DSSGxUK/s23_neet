import pandas as pd
from pandera.engines import pandas_engine

class YesNoBool(pandas_engine.BOOL):
    """
    An extension of the default boolean class to change the coerce behaviour.
    Inherits the pandera boolean class (which is a wrapper for the pandas one).

    Please see the explanation in the pandera documentation for further information:
    https://pandera.readthedocs.io/en/stable/dtypes.html
    """

    def coerce(self, series: pd.Series) -> pd.Series:
        """
        Coerce a pandas.Series to boolean types if column includes 
        different spellings for "Yes" and "No". If a column contains 
        a mix of e.g. "Yes", "Y", "no" and "No" will throw a Value Error.
        
        Args:
            self
            series: pd.Series of the column to coerce
            
        Returns:
            series: coerced pd.Series
            
        Raises: 
            ValueError: If the series as more than to unique values
        """

        # If dtype is already boolean we can return early
        if pd.api.types.is_bool_dtype(series):
            return series.astype("boolean")
        
        # First check if column contains only two values
        if len(series.dropna().unique()) > 2:
            raise ValueError("Column has more than two unique values (ignoring NaNs).")

        series = series.replace(
            {
                "Yes": 1,
                "YES":1,
                "Y": 1,
                "yes": 1,
                "y": 1,               
                "No": 0,
                "NO": 0,
                "N": 0,
                "no": 0,
                "n": 0,
            }
        )

        return series.astype("boolean")