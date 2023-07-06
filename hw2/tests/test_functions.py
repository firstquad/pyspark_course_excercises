import pytest
from pyspark.sql import functions as F, types
from chispa import *
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType
from video_analytics import functions as func


from video_analytics.functions import (
    split_tags_custom_udf,
    median,
    video_score
)


@pytest.fixture(scope='session')
def spark():
    return (
        SparkSession
        .builder
        .master("local")
        .appName("chispa")
        .getOrCreate()
    )


def test_score(spark):
    data = [
        (435, 0, 678, 5, 35, 111340.0),
        (6, 1, 6, 6, 7, 1313.0),
    ]
    df = spark.createDataFrame(data, ['views',
                                      'likes',
                                      'dislikes',
                                      'comment_likes',
                                      'comment_replies',
                                      'expected_score']) \
        .withColumn('score', func.video_score_udf('views',
                                                  'likes',
                                                  'dislikes',
                                                  'comment_likes',
                                                  'comment_replies'))
    assert_column_equality(df, 'score', 'expected_score')


def test_split_tags_custom_udf(spark):
    data = [
        ("cat|dog|food", ["cat", "dog", "food"]),
        ("", [""]),
        ("cat", ["cat"]),
        (None, None)
    ]

    df = (
        spark
        .createDataFrame(data, ["tags", "expected_tags"])
        .withColumn("parsed_tags", split_tags_custom_udf(F.col("tags")))
    )

    assert_column_equality(df, "parsed_tags", "expected_tags")



def test_median(spark):
    data = [x for x in range(1, 5)]

    rows = spark.createDataFrame(data, IntegerType()) \
        .agg(func.median('value').alias('median')) \
        .collect()

    assert rows[0]['median'] == 2.5
