import pyspark.sql.types as T
import pandas as pd
from pyspark.sql import functions as F, types

import numpy as np

def video_score(views: pd.Series, likes: pd.Series, dislikes: pd.Series, comment_likes: pd.Series, comment_replies: pd.Series) -> pd.Series:
    return 100*(views + likes + dislikes) + comment_likes + comment_replies

video_score_udf = F.pandas_udf(video_score,
                               returnType=types.DoubleType())

@F.pandas_udf(T.FloatType(), F.PandasUDFType.GROUPED_AGG)
def median(scores) -> float:
    return np.median(scores)


@F.pandas_udf(T.ArrayType(T.StringType(), True), F.PandasUDFType.SCALAR)
def split_tags_custom_udf(tags):
    return tags.str.split('|')


