import numpy as np
import pandas as pd
import urllib.request as request
import time
from pandas import DataFrame

import pyarrow.parquet as pq
import pyarrow as pa


class SimpleWordCountExample:
    """
    source file: https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
    save output file into output folder
    """

    def __init__(self):
        self.source_url = 'https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt'
        self.get_current_time_func = lambda: int(round(time.time() * 1000))

    def load_source_file(self) -> DataFrame:
        return pd.read_fwf(self.source_url)

    def do_word_count(self, df: DataFrame) -> DataFrame:
        return df.iloc[:,0].str \
               .extractall(pat=r"(\b[a-zA-Z0-9]+\b)")[0] \
               .value_counts() \
               .rename_axis('word') \
               .reset_index(name='count')

    def persist_result(self, df: DataFrame):
        df['start_with']=df['word'].map(lambda x: np.where(x.isalpha(), x.capitalize()[0:1] ,"others"))
        print(df)

        # df.to_csv('words_count.csv', index=False)
        # df.to_parquet('words_count.parquet', index=False)

        adf = pa.Table.from_pandas(df)
        pq.write_to_dataset(adf, root_path='dataset',partition_cols=['start_with'])



    def run(self):
        raw_df = self.load_source_file()
        wordcount_result_df = self.do_word_count(raw_df)
        self.persist_result(wordcount_result_df)


if __name__ == "__main__":
    runner = SimpleWordCountExample()
    runner.run()
