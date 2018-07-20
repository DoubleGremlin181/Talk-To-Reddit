import sqlite3
import pandas as pd

timeframe = ['2018-01', '2018-02','2018-03', '2018-04', '2018-05']


connection = sqlite3.connect('Comment_Dataset.db')
c = connection.cursor()
sql = "SELECT COUNT(*) FROM comment_data WHERE PARENT NOT NULL"
c.execute(sql)
limit = c.fetchone()[0]
print ("The number of training pairs are: " + str(limit))
df = pd.read_sql("SELECT * FROM comment_data WHERE parent NOT NULL and score > 0 ORDER BY unix ASC", connection)
cur_length = len(df)
with open('train.from', 'a', encoding = 'UTF-8') as f:
    for content in df['parent'].values:
        f.write(content+'\n')
with open('train.to', 'a', encoding = 'UTF-8') as f:
    for content in df['comment'].values:
        f.write(str(content)+'\n')
