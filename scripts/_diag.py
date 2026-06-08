import duckdb, os
from dotenv import load_dotenv
import pathlib
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
df = con.execute(f"SELECT * FROM read_parquet('s3://{bucket}/history/fundamentals/data.parquet')").df()
import pandas as pd
print("rows:", len(df), "| RICs:", df["ric"].nunique())
print("date range:", df["date"].min(), "->", df["date"].max())
print()
for c in ["pb_ratio", "market_cap", "shares_outstanding", "gics_sector"]:
    pct = df[c].isna().mean()
    sample = df[c].dropna().iloc[:3].tolist()
    print(f"  {c}: {pct:.1%} null | sample: {sample}")
print()
print("gics_sector value_counts:")
print(df["gics_sector"].value_counts(dropna=False).head(15).to_string())
latest = df[df["date"] == df["date"].max()]
print()
print("Latest snapshot RICs:", latest["ric"].nunique())
print("Latest mcap range:", latest["market_cap"].min(), "->", latest["market_cap"].max())
print("Latest shares range:", latest["shares_outstanding"].min(), "->", latest["shares_outstanding"].max())
