# import psycopg, os

# conn = psycopg.connect(os.environ["PG_DSN"])
# cur = conn.cursor()
# cur.execute("SELECT version();")
# print(cur.fetchone())
# conn.close()


import pandas as pd

df = pd.read_csv("data/two_games.csv")
print(len(df.columns))