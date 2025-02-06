import mysql.connector
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine
from datetime import datetime, timedelta

#creating mysql connection
def get_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Aher@123",
            db="stock_data"

        )
        if connection.is_connected():
            print("Connected to MySQL Database")
            return connection
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None
        

# create a database 
def create_database_ifnot_exists(cursor,db):
    query =f"show database name {db}"
    cursor.execute(query)

    result=cursor.fetchall()
    if db not in [row[0] for row in result]:
        query = f"CREATE DATABASE {db}"
        cursor.execute(query)
        print(f"Database {db} created")
    else:
        print(f"Database {db} already exists")


def create_table_ifnot_exists(cursor,db,table):
    query = f"show tables from {db}.{table}"
    cursor.execute(query)
    result=cursor.fetchall()
    if table not in [row[0] for row in result]:
        query = f""" CREATE TABLE {db}.{table} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        date DATE,
        open float,
        high float,
        low float,
        close float,
        volume INT
        )"""
        cursor.execute(query)
        print(f"Table {table} created")
    else:
        print(f"Table {table} already exists")


#fetch historical data
def fetch_historical_data(symbol):
    try:
        if symbol:
            histdata=yf.download(symbol)
            histdata.reset_index(inplace=True)
            print(f"symbol data is bieing fetched for {symbol}")
            return histdata
        else:
            print("Symbol is not provided.")
            return None
    except error as e:
        print("Error fetching historical data",e)

def fetch_livedata(symbol,api_key):
    api_key="771O3VPDZ5UH78E3"
    try:
        if symbol and api_key:
            url = 'https://www.alphavantage.co/query'
            params={
                "function":"TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": api_key,
                "outputsize": "full"
            }
            response=requests.get(url,params=params)
            livedata=response.json()
            print(f"live data is being fetched for {symbol}")

            #check for errors
            if "Time Series (Daily)" not in livedata:
                print("Error fetching live data")
                return pd.DataFrame()
            
            daily_data = livedata["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(daily_data, orient='index')
            df.reset_index(inplace=True)
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date', ascending=True)
            return df
        else:
            print("Symbol or API key is not provided.")
            return None
    except Exception as e:
        print("Error fetching live data:", e)

def insert_data_insert_into_table(cursor,db,table,data):
        cursor.excute("use {db}")
        df=pd.DataFrame(data)
        df.to_sql(table,db,if_exists='replace',index=False)

        df.drop_duplicates(subset=['timestamp'],keep='last',inplace=True)

        for _,row in df.iterrows():
           query=f"""
                    Insert into {table}(timestamp,open,high,low,close,volume)
                    values(%s,%s,%s,%s,%s,%s)
                    """
           cursor.execute(query,(row['timestamp'],row['open'],row['high'],row['low'],row['volume']))
        print(f"{len(df)} rows inserted into {table}.")

def run_pipeline(db,symbol):
    # Connect to the database
    conn=get_mysql_connection()
    cursor=conn.cursor()

    create_database_ifnot_exists(cursor,db)
    create_table_ifnot_exists(cursor,db,symbol)
    historical_data=fetch_historical_data(symbol)
    if historical_data.empty:
        print("No historical data available for the symbol")
    else:
        insert_data_insert_into_table(cursor,db,symbol,historical_data)

    livedata=fetch_livedata(symbol)  
    if livedata.empty:
        print("No live data available for the symbol")
    else:
        insert_data_insert_into_table(cursor,db,symbol,livedata)

    conn.commit()    
    query = f"SELECT * FROM {db_name}.{ticker}"
    df = pd.read_sql(query, conn)

    print(df.head())  # Show a preview of the data
    cursor.close()
    conn.close()


run_pipeline(db ="stock_data",symbol=input("Enter the symbol: "))











