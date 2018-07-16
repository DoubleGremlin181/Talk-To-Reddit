import sqlite3
import json
from datetime import datetime


timeframe = ['2018-01', '2018-02','2018-03', '2018-04', '2018-05']

sql_transaction = []
connection = sqlite3.connect('Comment_Dataset.db')
c = connection.cursor()


def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS comment_data(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")

def format_data(data):
    data=data.replace('\n', '  newlinechar  ').replace('\r', '  newlinechar  ').replace('"', "'")
    return data

def find_parent(pid):
    try:
        sql = "SELECT comment FROM comment_data WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        #print(str(e))
        return False

def find_existing_score(pid):
    try:
        sql = "SELECT score FROM comment_data WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        #print(str(e))
        return False

def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) <  1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[removed]':
        return False
    elif data == '[deleted]':
        return False
    else:
        return True

def sql_insert_replace_comment(comment_id, parent_id, parent_data, body, subreddit, created_utc, score):
    try:
        sql = "UPDATE comment_data SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id = ?".format(parent_id, comment_id, parent_data, body, subreddit, int(created_utc), score, parent_id)
        transaction_bldr(sql)
    except Exception as e:
        print ('s0 insertion', str(e))

def sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_utc, score):
    try:
        sql = """INSERT INTO comment_data (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}", "{}", "{}", "{}", "{}", {}, {})""".format(parent_id, comment_id, parent_data, body, subreddit, int(created_utc), score)
        transaction_bldr(sql)
    except Exception as e:
        print ('s0 insertion', str(e))

def sql_insert_no_parent(comment_id, parent_id, body, subreddit, created_utc, score):
    try:
        sql = """INSERT INTO comment_data (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}", "{}", "{}", "{}", {}, {})""".format(parent_id, comment_id, body, subreddit, int(created_utc), score)
        transaction_bldr(sql)
    except Exception as e:
        print ('s0 insertion', str(e))

def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []

if __name__ == '__main__':
    print("Operation started:")
    create_table()
    row_counter = 0
    paired_rows = 0
    for i in timeframe:
        print("Month: {}".format(i))
        with open ('/run/media/kavish/New Volume/Projects/Datasets/RC_{}'.format(i), buffering=1000) as f:
            for row in f:
                row_counter += 1
                row = json.loads(row)
                parent_id = row['parent_id']
                body = format_data(row['body'])
                created_utc = row['created_utc']
                score = row['score']
                comment_id = row['link_id']
                subreddit = row['subreddit']
                parent_data = find_parent(parent_id)

                if score >= 2:
                    existing_comment_score = find_existing_score(parent_id)
                    if existing_comment_score:
                        if score > existing_comment_score:
                            if acceptable(body):
                                sql_insert_replace_comment(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                    else:
                        if acceptable(body):
                            if parent_data:
                                sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                                paired_rows += 1
                            else:
                                sql_insert_no_parent(comment_id, parent_id, body, subreddit, created_utc, score)
                if row_counter % 100000 == 0:
                    print('Total Rows Read: {}, Paired Rows {}, Time {}'.format(row_counter, paired_rows, str(datetime.now())))
