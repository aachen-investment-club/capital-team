import sys, pathlib, os, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
import duckdb, boto3

bucket = os.getenv("S3_BUCKET", "")
region = os.getenv("AWS_REGION", "eu-central-1")
creds  = boto3.Session().get_credentials().get_frozen_credentials()
con    = duckdb.connect(":memory:")
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"SET s3_region='{region}';")
con.execute(f"SET s3_access_key_id='{creds.access_key}';")
con.execute(f"SET s3_secret_access_key='{creds.secret_key}';")
df = con.execute(f"SELECT * FROM read_parquet('s3://{bucket}/history/fundamentals/data.parquet')").df()
df["date"] = pd.to_datetime(df["date"])

ric = df["ric"].iloc[0]
sub = df[df["ric"] == ric].sort_values("date")
as_of = pd.Timestamp("2026-06-07")
d12m  = as_of - pd.DateOffset(months=12)
print(f"Sample RIC: {ric}")
print(f"Date range: {sub['date'].min().date()} -> {sub['date'].max().date()}")
print(f"12m lookback target: {d12m.date()}")
print(f"Rows at/before 12m date: {len(sub[sub['date'] <= d12m])}")
print(f"Earliest available date: {sub['date'].min().date()}")
print(f"Gap to 12m: {(d12m - sub['date'].min()).days} days")
print()
print(f"Total unique RICs in fundamentals: {df['ric'].nunique()}")
