import sqlite3
import pandas as pd

timeframe = ['2018-01', '2018-02','2018-03', '2018-04', '2018-05']

for month in timeframe:
    connection = sqlite3.connect('Comment_Dataset.db')
    c = connection.cursor()
    limit = 5000
    last_unix = 0
    cur_length = limit
    counter = 0
    test_done = False

    while cur_length == limit:

        df = pd.read_sql("SELECT * FROM comment_data WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix, limit), connection)
        last_unix = df.tail(1)['unix'].values[0]
        cur_length = len(df)
        if not test_done:
            with open('test.from', 'a', encoding = 'UTF-8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')
            with open('test.to', 'a', encoding = 'UTF-8') as f:
                for content in df['comment'].values:
                    f.write(str(content)+'\n')
        else:
            with open('train.from', 'a', encoding = 'UTF-8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')
            with open('train.to', 'a', encoding = 'UTF-8') as f:
                for content in df['comment'].values:
                    f.write(str(content)+'\n')
        counter +=1
        if counter % 20 == 0:
            print(counter*limit, 'rows compiled so far')
