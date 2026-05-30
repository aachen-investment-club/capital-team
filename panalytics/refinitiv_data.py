import os
from datetime import date, timedelta
from dotenv import load_dotenv
import eikon as ek
import pandas as pd

load_dotenv()

ek.set_app_key(os.environ.get("refinitiv_api_key"))

# date range from one year ago to yesterday
def get_date_range() -> tuple[str, str]:
    today = date.today()
    yesterday = today - timedelta(days=1)
    one_year_ago = today - timedelta(days=365)
    return one_year_ago.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")


def get_timeseries(rics: list[str], fields: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    # Fetch one RIC at a time — Eikon limits total data points per request,
    # so batching multiple RICs silently truncates the date range from the start.
    results = []
    for i, ric in enumerate(rics):
        print(f"Fetching {ric} ({i + 1}/{len(rics)})...")
        df = ek.get_timeseries(ric, fields=fields, start_date=start_date, end_date=end_date)
        if isinstance(df, pd.DataFrame):
            df.columns = pd.MultiIndex.from_product([[ric], df.columns])
            results.append(df)
        else:
            print(f"{ric} returned no data")
    return pd.concat(results, axis=1)


def save_to_csv(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath)
    print(f"Saved to {filepath}")

# not working (some RICs are wrong)
sp500_rics = [
    "MMM.N", "AOS.N", "ABT.N", "ABBV.N", "ACGL.O", "ACN.N", "ADBE.O", "ADM.N", "ADP.O", "ADSK.O",
    "AEE.N", "AEP.O", "AES.N", "AFL.N", "A.N", "APD.N", "ABNB.O", "AKAM.O", "ALB.N", "ARE.N",
    "ALGN.O", "ALLE.N", "LNT.O", "ALL.N", "GOOGL.O", "GOOG.O", "MO.N", "AMZN.O", "AMCR.N", "AMD.O",
    "AIG.N", "AMT.N", "AWK.N", "AMP.N", "AME.N", "AMGN.O", "APH.N", "ADI.O", "ANSS.O", "AON.N",
    "APA.O", "APTV.N", "AJG.N", "AIZ.N", "ATO.N", "T.N", "AZO.N", "AVB.N", "AVY.N", "AXON.O",
    "BKR.O", "BALL.N", "BAC.N", "BBWI.N", "BAX.N", "BDX.N", "BRKb.N", "BBY.N", "BIO.N", "TECH.O",
    "BLK.N", "BX.N", "BA.N", "BKNG.O", "BWA.N", "BXP.N", "BSX.N", "BMY.N", "AVGO.O", "BR.N",
    "BRO.N", "BLDR.N", "BG.N", "CHRW.O", "CDNS.O", "CZR.O", "CPT.N", "CPB.N", "COF.N", "CAH.N",
    "KMX.N", "CCL.N", "CARR.N", "CAT.N", "CBOE.O", "CBRE.N", "CDW.O", "CE.N", "COR.N", "CNC.N",
    "CF.N", "CRL.N", "SCHW.N", "CHTR.O", "CVX.N", "CMG.N", "CB.N", "CHD.N", "CI.N", "CINF.O",
    "CTAS.O", "CSCO.O", "C.N", "CFG.N", "CLX.N", "CME.O", "CMS.N", "KO.N", "CTSH.O", "CL.N",
    "CMCSA.O", "CMA.N", "CAG.N", "COP.N", "ED.N", "STZ.N", "COO.N", "CPRT.O", "GLW.N", "CTVA.N",
    "CSGP.O", "COST.O", "CTRA.O", "CCI.N", "CSX.O", "CMI.N", "CVS.N", "DHI.N", "DHR.N", "DRI.N",
    "DVA.N", "DAY.N", "DECK.N", "DE.N", "DAL.N", "DVN.N", "DXCM.O", "FANG.O", "DLR.N", "DFS.N",
    "DG.N", "DLTR.O", "D.N", "DPZ.N", "DOV.N", "DOW.N", "DTE.N", "DUK.N", "DD.N", "EMN.N", "ETN.N",
    "EBAY.O", "ECL.N", "EIX.N", "EW.N", "EA.O", "ELV.N", "LLY.N", "EMR.N", "ENPH.O", "ETR.N",
    "EOG.N", "EPAM.N", "EQT.N", "EFX.N", "EQIX.O", "EQR.N", "ESS.N", "EL.N", "ETSY.O", "EG.N",
    "EVRG.O", "ES.N", "EXC.O", "EXPE.O", "EXPD.O", "EXR.N", "XOM.N", "FFIV.O", "FDS.O", "FICO.N",
    "FAST.O", "FRT.N", "FDX.N", "FIS.N", "FITB.O", "FSLR.O", "FE.N", "FI.N", "F.N", "FTNT.O",
    "FTV.N", "FOXA.O", "FOX.O", "BEN.N", "FCX.N", "GRMN.O", "IT.N", "GE.N", "GEHC.O", "GEN.O",
    "GNRC.N", "GD.N", "GIS.N", "GPC.N", "GILD.O", "GS.N", "HAL.N", "HIG.N", "HAS.O", "HCA.N",
    "HSIC.O", "HSY.N", "HES.N", "HPE.N", "HLT.N", "HOLX.O", "HD.N", "HON.N", "HRL.N", "HST.N",
    "HWM.N", "HPQ.N", "HUBB.N", "HUM.N", "HBAN.O", "HII.N", "IBM.N", "IEX.N", "IDXX.O", "ITW.N",
    "ILMN.O", "INCY.O", "IR.N", "PODD.O", "INTC.O", "ICE.N", "IFF.N", "IP.N", "IPG.N", "INTU.O",
    "ISRG.O", "IVZ.N", "INVH.N", "IQV.N", "IRM.N", "JBHT.O", "JBL.N", "JKHY.O", "J.N", "JNJ.N",
    "JCI.N", "JPM.N", "JNPR.N", "K.N", "KDP.O", "KEY.N", "KEYS.N", "KMB.N", "KIM.N", "KMI.N",
    "KLAC.O", "KHC.O", "KR.N", "LHX.N", "LH.N", "LRCX.O", "LW.N", "LVS.N", "LDOS.N", "LEN.N",
    "LNC.N", "LIN.N", "LYV.N", "LKQ.O", "LMT.N", "L.N", "LOW.N", "LYB.N", "MTB.N", "MRO.N",
    "MPC.N", "MKTX.O", "MAR.O", "MMC.N", "MLM.N", "MAS.N", "MA.N", "MTCH.O", "MKC.N", "MCD.N",
    "MCK.N", "MDT.N", "MRK.N", "META.O", "MET.N", "MTD.N", "MGM.N", "MCHP.O", "MU.O", "MSFT.O",
    "MAA.N", "MRNA.O", "MHK.N", "MOH.N", "TAP.N", "MDLZ.O", "MPWR.O", "MNST.O", "MCO.N", "MS.N",
    "MOS.N", "MSI.N", "MSCI.N", "NDAQ.O", "NTAP.O", "NWSA.O", "NWS.O", "NEE.N", "NKE.N", "NI.N",
    "NFLX.O", "NWL.O", "NRG.N", "NSC.N", "NTRS.O", "NOC.N", "NCLH.N", "NUE.N", "NVDA.O", "NVR.N",
    "NXPI.O", "ORLY.O", "OXY.N", "ODFL.O", "OMC.N", "ON.O", "OKE.N", "ORCL.N", "OTIS.N", "PCAR.O",
    "PKG.N", "PANW.O", "PARA.O", "PH.N", "PAYX.O", "PAYC.N", "PYPL.O", "PNR.N", "PEP.O", "PFE.N",
    "PCG.N", "PM.N", "PSX.N", "PNW.O", "PNC.N", "POOL.O", "PPG.N", "PPL.N", "PFG.O", "PG.N",
    "PGR.N", "PLD.N", "PRU.N", "PEG.N", "PTC.O", "PSA.N", "PHM.N", "PWR.N", "QCOM.O", "DGX.N",
    "RL.N", "RJF.N", "RTX.N", "O.N", "REG.O", "REGN.O", "RF.N", "RSG.N", "RMD.N", "RVTY.N",
    "RHI.N", "ROK.N", "ROL.N", "ROP.N", "ROST.O", "RCL.N", "SPGI.N", "CRM.N", "SBAC.O", "SLB.N",
    "STX.O", "SEE.N", "SRE.N", "NOW.N", "SHW.N", "SPG.N", "SWKS.O", "SJM.N", "SNA.N", "SO.N",
    "LUV.N", "SWK.N", "SBUX.O", "STT.N", "STLD.O", "STE.N", "SYK.N", "SYF.N", "SNPS.O", "SYY.N",
    "TMUS.O", "TROW.O", "TTWO.O", "TPR.N", "TGT.N", "TEL.N", "TDY.N", "TFX.N", "TER.O", "TSLA.O",
    "TXN.O", "TXT.N", "TMO.N", "TJX.N", "TSCO.O", "TT.N", "TDG.N", "TRV.N", "TRMB.O", "TFC.N",
    "TYL.N", "TSN.N", "USB.N", "UDR.N", "ULTA.O", "UNP.N", "UAL.O", "UPS.N", "URI.N", "UNH.N",
    "UHS.N", "VLO.N", "VTR.N", "VRSN.O", "VRSK.O", "VZ.N", "VRTX.O", "VTRS.O", "VICI.N", "V.N",
    "VMC.N", "WRB.N", "GWW.N", "WAB.N", "WBA.O", "WMT.N", "WBD.O", "WDAY.O", "WEC.N", "WFC.N",
    "WELL.N", "WDC.O", "WST.N", "WEX.N", "WY.N", "WYNN.O", "XEL.O", "XYL.N", "YUM.N", "ZBRA.O",
    "ZBH.N", "ZTS.N", "ZM.O", "QRVO.O", "LULU.O", "ANET.N", "SMCI.O", "APP.O", "SOLV.N", "DOC.N"
]

# has to adjust csv file bc 3 first rows are not the desired format
if __name__ == "__main__":
    start_date, end_date = get_date_range()
    ts_df = get_timeseries(
        rics=[
            "AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "TSLA.O", "META.O", "NVDA.O", "JPM.N", "V.N", "UNH.N", ".MIWO00000PUS"
        ],
        fields=["CLOSE"],
        start_date=start_date,
        end_date=end_date, # 252 trading days in a year
    )
    save_to_csv(ts_df, "data/timeseries.csv")
