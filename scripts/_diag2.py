import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import os, duckdb, pandas as pd, numpy as np
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

bucket = os.getenv("S3_BUCKET", "")
region = os.getenv("AWS_REGION", "eu-central-1")
key    = os.getenv("AWS_ACCESS_KEY_ID", "")
secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")

con = duckdb.connect(":memory:")
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"SET s3_region='{region}';")
con.execute(f"SET s3_access_key_id='{key}';")
con.execute(f"SET s3_secret_access_key='{secret}';")

fund = con.execute(f"SELECT * FROM read_parquet('s3://{bucket}/history/fundamentals/data.parquet')").df()
fund["date"] = pd.to_datetime(fund["date"])
latest_date = fund["date"].max()
snap = fund[fund["date"] == latest_date]

print(f"Snapshot date: {latest_date.date()}  |  {len(snap)} securities")
print(f"\nmarket_cap null: {snap['market_cap'].isna().sum()}/{len(snap)}")
print(f"market_cap sample: {snap['market_cap'].dropna().head(5).tolist()}")
print()

# Check which EOD parquet files exist for universe RICs
import boto3
s3 = boto3.client("s3", region_name=region)
resp = s3.list_objects_v2(Bucket=bucket, Prefix="history/eod_prices/")
eod_keys = [o["Key"] for o in resp.get("Contents", [])]
print(f"EOD parquet files in S3: {len(eod_keys)}")
for k in eod_keys[:10]:
    print(f"  {k}")

# Check security master
sm = pd.read_csv(pathlib.Path(__file__).parent.parent / "config" / "security_master.csv", dtype=str)
sm["active"] = sm["active"].str.lower().isin(("true","1","yes"))
sm = sm[sm["active"]]
print(f"\nSecurity master active: {len(sm)}")
print(sm[["security_id","ric","ticker","asset_type"]].to_string())

# How many universe RICs have EOD data?
eod_ids = {k.split("security_id=")[1].split("/")[0] for k in eod_keys if "security_id=" in k}
sm_with_eod = sm[sm["security_id"].isin(eod_ids)]
print(f"\nSecurity master entries with EOD data: {len(sm_with_eod)}")
print(sm_with_eod[["security_id","ric","ticker"]].to_string())
