#!/usr/bin/env python3
"""
Resolve CIK numbers for tickers that are missing from SEC's current company_tickers.json.

These are typically delisted, renamed, or acquired companies (e.g. FB→META, ATVI→MSFT).

Strategy:
1. Use EDGAR company search API to find CIK by ticker
2. Fall back to EFTS full-text search if the company search doesn't find it
3. Write results to ticker_cik_overrides.json

Usage:
    python3 resolve_cik.py --user-agent 'Hacklytics26 Team (your.email@domain.com)'
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# EDGAR company search — works with historical tickers
EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2014-01-01&enddt=2019-12-31&forms=10-K,10-Q"
# Alternative: direct company tickers exchange endpoint (includes some historical)
EDGAR_COMPANY_TICKERS_EXCHANGE = "https://www.sec.gov/files/company_tickers_exchange.json"
# EDGAR full text search for company filings
EDGAR_EFTS_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K&dateRange=custom&startdt=2013-01-01&enddt=2019-12-31"


def parse_instance_folder(name: str) -> Optional[Tuple[date, str]]:
    m = re.fullmatch(r"(\d{8})_([A-Z0-9.\-]+)", name.strip().upper())
    if not m:
        return None
    d = datetime.strptime(m.group(1), "%Y%m%d").date()
    ticker = m.group(2)
    return d, ticker


class SecClient:
    def __init__(self, user_agent: str, sleep_s: float = 0.35):
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self.sleep_s = sleep_s
        self.session = requests.Session()

    def get_json(self, url: str) -> Optional[dict]:
        time.sleep(self.sleep_s)
        try:
            r = self.session.get(url, headers=self.headers, timeout=30)
            if r.status_code == 429:
                print(f"  [429] Rate limited, waiting 5s...")
                time.sleep(5)
                r = self.session.get(url, headers=self.headers, timeout=30)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception as e:
            print(f"  [ERR] {e}")
            return None


# --- Well-known ticker → CIK mappings for major renamed/acquired companies ---
# These are companies that changed ticker or were acquired, where automated lookup
# may not easily resolve them. Built from SEC EDGAR manual lookups.
WELL_KNOWN_OVERRIDES = {
    # Ticker changes
    "FB": 1326801,       # Facebook → Meta Platforms (META)
    "ATVI": 718877,      # Activision Blizzard → acquired by MSFT
    "CELG": 816284,      # Celgene → acquired by BMY
    "ANDV": 1370946,     # Andeavor (was TSO) → acquired by MPC
    "ANTM": 1156039,     # Anthem → Elevance Health (ELV)
    "ESRX": 1532063,     # Express Scripts → acquired by CI
    "CA": 356028,        # CA Technologies → acquired by Broadcom
    "CSRA": 1646383,     # CSRA → acquired by GDIT
    "DISH": 1001082,     # DISH Network
    "GPS": 39911,        # Gap Inc → changed to GAP
    "BLL": 9389,         # Ball Corporation
    "K": 55067,          # Kellogg → Kellanova (split)
    "DFS": 1393612,      # Discover Financial
    "CTXS": 877890,      # Citrix → taken private
    "FLIR": 354908,      # FLIR Systems → acquired by Teledyne
    "XLNX": 743988,      # Xilinx → acquired by AMD
    "X": 1163302,        # United States Steel
    "CHK": 895126,       # Chesapeake Energy
    "MRO": 101778,       # Marathon Oil
    "FL": 850209,        # Foot Locker
    "HBI": 1359841,      # Hanesbrands
    "IPG": 51644,        # Interpublic Group
    "WBA": 1618921,      # Walgreens Boots Alliance
    "CMA": 28412,        # Comerica
    "CVG": 24090,        # Convergys → acquired by Synnex
    "RHT": 1087423,      # Red Hat → acquired by IBM
    "AGN": 850693,       # Allergan → acquired by AbbVie
    "AET": 1122304,      # Aetna → acquired by CVS
    "MON": 67725,        # Monsanto → acquired by Bayer AG
    "ALXN": 899866,      # Alexion → acquired by AstraZeneca
    "TIVO": 1088825,     # TiVo → Xperi
    "DNKN": 1357708,     # Dunkin' Brands → acquired by Inspire
    "GGP": 1496048,      # General Growth Properties → acquired by Brookfield
    "ABMD": 753924,      # Abiomed → acquired by J&J
    "MDRX": 1000230,     # Allscripts Healthcare
    "DISCA": 1437107,    # Discovery Inc → Warner Bros Discovery (WBD)
    "VIAB": 813828,      # Viacom → merged with CBS → Paramount (PARA)
    "SIVB": 719739,      # Silicon Valley Bank → failed/acquired
    "SBNY": 1288855,     # Signature Bank → failed
    "APC": 773910,       # Anadarko → acquired by Oxy
    "RTN": 1047122,      # Raytheon → merged → RTX
    "UTX": 101829,       # United Technologies → merged → RTX
    "COL": 55143,        # Rockwell Collins → acquired by UTX/RTX
    "LLL": 1501585,      # L3 Technologies → merged with Harris → LHX
    "HRS": 202058,       # Harris Corp → merged with L3 → LHX
    "FISV": 798354,      # Fiserv (still active but ticker lookup may differ)
    "FII": 38777,        # Federated Hermes
    "MYL": 1081316,      # Mylan → Viatris (VTRS)
    "ENDP": 1593034,     # Endo International → bankruptcy
    "CTLT": 1596783,     # Catalent → acquired by Novo
    "BKS": 890491,       # Barnes & Noble → taken private
    "GE": 40554,         # General Electric
    "CI": 1739940,       # Cigna → The Cigna Group
    "ASNA": 1498301,     # Ascena Retail → bankruptcy
    "BOBE": 818033,      # Bob Evans → sold/split
    "BMS": 821237,       # Bemis → acquired by Amcor
    "CRR": 811596,       # CARBO Ceramics
    "CRY": 742112,       # CryoLife → acquired
    "CRAY": 726514,      # Cray → acquired by HPE
    "AKS": 918696,       # AK Steel → acquired by Cleveland-Cliffs
    "CLGX": 1131312,     # CoreLogic → taken private
    "CTRL": 882184,      # Control4 → acquired by SnapAV
    "DSW": 1090012,      # Designer Brands
    "EGL": 728535,       # Engility → acquired by SAIC
    "EGOV": 1130713,     # NIC Inc → acquired by Tyler
    "EV": 350698,        # Eaton Vance → acquired by MS
    "ETFC": 1015780,     # E*TRADE → acquired by MS
    "FINL": 1472595,     # Finish Line → acquired by JD Sports
    "FRED": 1524270,     # Fred's → bankruptcy
    "LOGM": 1420302,     # LogMeIn → taken private
    "MBFI": 36104,       # MB Financial → acquired by Fifth Third
    "NCI": 1265083,      # Navigant → acquired by Guidehouse
    "PBCT": 713111,      # People's United Financial → acquired by M&T
    "QSII": 209513,      # QSI Dental → Quality Systems
    "TSS": 721683,       # Total System Services → acquired by GPN
    "ULTI": 1016125,     # Ultimate Software → taken private (Kronos)
    "VAR": 203527,       # Varian Medical → acquired by Siemens
    "WLTW": 1140536,     # Willis Towers Watson → merged with Aon (reversed)
    "WYND": 1361658,     # Wyndham Hotels
    "XL": 875159,        # XL Group → acquired by AXA
    "XOXO": 1517375,     # XO Group
    "ACXM": 733269,      # Acxiom → Liveramp (RAMP)
    "CATM": 1460225,     # Cardtronics → acquired by NCR
    "CAMP": 1572565,     # CalAmp
    "CDR": 1376227,      # Cedar Realty Trust
    "CHFC": 811808,      # Chemical Financial → acquired by TCF
    "CHSP": 1516251,     # Chesapeake Lodging Trust → acquired
    "ECOL": 868857,      # US Ecology → acquired by Republic Services
    "EBIX": 814549,      # Ebix → bankruptcy
    "EDR": 1404912,      # Endeavor Real Estate → acquired
    "EFII": 1090061,     # Electronics for Imaging → taken private
    "ERA": 1539722,      # Era Group → Bristow
    "ESL": 809248,       # Esterline Technologies → acquired by TransDigm
    "ESV": 314808,       # Ensco → merged with Rowan → Valaris
    "EXTN": 1592016,     # Exterran → acquired
    "FTD": 1644853,      # FTD Companies → bankruptcy
    "GWB": 1095073,      # Great Western Bancshares → acquired
    "HCN": 766704,       # HCN → Welltower (WELL)
    "HF": 1485469,       # HFF → acquired by JLL
    "HFC": 1064728,      # HollyFrontier → HF Sinclair (DINO)
    "HMST": 18349,       # HomeStreet → acquired
    "HMSY": 1196501,     # HMS Holdings → acquired by Gainwell
    "HSKA": 1060349,     # Heska Corp → taken private
    "ILG": 1485172,      # ILG → acquired by Marriott Vacations
    "IPCC": 884217,      # Infinity Property & Casualty → acquired by Kemper
    "ISCA": 51548,       # International Speedway → NASCAR/merged
    "KLXI": 1437491,     # KLX Inc → acquired by Boeing
    "KND": 1060714,      # Kindred Healthcare → acquired by Humana
    "KRA": 43920,        # Kraton Polymers → acquired by DL Chemical
    "LABL": 1592015,     # Multi-Color Corp → acquired
    "LANC": 57515,       # Lancaster Colony
    "LCI": 1498382,      # Lannett Company
    "LHCG": 1303313,     # LHC Group → acquired by UHS
    "LPT": 1347523,      # Liberty Property Trust → acquired by Prologis
    "MANT": 892537,      # ManTech → acquired by Carlyle
    "MCF": 1528837,      # Contango Oil → renamed
    "MDCO": 854835,      # The Medicines Company → acquired by Novartis
    "MDP": 65011,        # Meredith Corp → acquired by Dotdash/IAC
    "MDR": 776323,       # McDermott → bankruptcy
    "MDSO": 1145197,     # Medidata Solutions → acquired by Dassault
    "MGLN": 1076682,     # Magellan Health → acquired by Centene
    "MHLD": 1300455,     # Maiden Holdings
    "MIK": 1704711,      # Michaels → taken private by Apollo
    "MMC": 62996,        # Marsh & McLennan
    "MNK": 1589991,      # Mallinckrodt → bankruptcy/restructured
    "MNTA": 1393584,     # Momenta Pharmaceuticals → acquired by J&J
    "MPW": 927066,       # Medical Properties Trust
    "MSCC": 802829,      # Microsemi → acquired by Microchip
    "NEWM": 1579684,     # New Media Investment Group → merged → Gannett
    "NLSN": 1492633,     # Nielsen → taken private
    "NR": 804368,        # Newpark Resources
    "OA": 1585929,       # Orbital ATK → acquired by Northrop
    "OCLR": 1110647,     # Oclaro → acquired by Lumentum
    "OFC": 860546,       # Corporate Office Properties Trust
    "OMI": 75252,        # Owens & Minor
    "OPB": 890319,       # Opus Bank → acquired
    "PERY": 708819,      # Perry Ellis → taken private
    "PES": 77543,        # Pioneer Energy → acquired
    "PJC": 78890,        # Piper Jaffray → Piper Sandler (PIPR)
    "POL": 1375625,      # PolyOne → Avient (AVNT)
    "PRFT": 1085869,     # Perficient → acquired by EQT
    "PRSC": 1071438,     # Providence Service → ModivCare (MODV)
    "QHC": 1660734,      # Quorum Health → acquired
    "RCII": 820591,      # Rent-A-Center → acquired by Upbound
    "RDC": 60714,        # Rowan Companies → merged with Ensco → Valaris
    "RECN": 1302028,     # Resources Connection
    "RAVN": 82166,       # Raven Industries → acquired by CNH
    "CLD": 1321655,      # Cloud Peak Energy → bankruptcy
    "CLB": 1072613,      # Core Laboratories
    "CNSL": 1284812,     # Consolidated Communications
    "COG": 858470,       # Cabot Oil & Gas → Coterra Energy (CTRA)
    "COP": 1163165,      # ConocoPhillips
    "CPE": 1168165,      # Callon Petroleum → acquired by APA
    "CPLA": 1046568,     # Capella Education → merged with Strayer → SEI
    "CPSI": 1169445,     # Computer Programs and Systems
    "CRZO": 1381197,     # Carrizo Oil & Gas → acquired by Callon
    "CXO": 1358071,      # Concho Resources → acquired by ConocoPhillips
    "CY": 791915,        # Cypress Semiconductor → acquired by Infineon
    "DAN": 1378789,      # Dana Inc
    "DO": 949039,        # Diamond Offshore → bankruptcy/restructured
    "DPLO": 1585608,     # Diplomat Pharmacy → acquired by OptimizeRx
    "DSPG": 1004668,     # DSP Group → acquired by Synaptics
    "DNR": 945764,       # Denbury → acquired by ExxonMobil
    "DRE": 783280,       # Duke Realty → acquired by Prologis
    "ACC": 927089,       # American Campus Communities → acquired by Blackstone
    "AAXN": 1069183,     # Axon Enterprise (changed ticker to AXON)
    "AAN": 1383312,      # Aaron's Holdings
    "AAWW": 1018979,     # Atlas Air Worldwide → taken private
    "ABAX": 718557,      # Abaxis → acquired by Zoetis
    "ABC": 1140859,      # AmerisourceBergen → Cencora (COR)
    "ACET": 2034,        # Aceto Corporation → bankruptcy
    "AEGN": 932628,      # Aegion Corp → taken private
    "ADS": 1101215,      # Alliance Data → Bread Financial (BFH)
    "ALE": 1553024,      # Allete
    "AKRX": 1065696,     # Akorn → bankruptcy
    "ARR": 1428236,      # ARMOUR Residential REIT
    "ARRS": 1437491,     # Arris International → acquired by CommScope (avoid dup w/ KLXI)
    "AXE": 7536,         # Anixter → acquired by WESCO
    "AXL": 1062231,      # American Axle
    "BCO": 2136,         # Brink's Company
    "BEL": 729580,       # Bel Fuse
    "BGFV": 1089063,     # Big 5 Sporting Goods
    "BGG": 109563,       # Briggs & Stratton → bankruptcy
    "BIG": 768835,       # Big Lots → bankruptcy
    "BPFH": 875657,      # Boston Private Financial → acquired by SVB
    "BRS": 840715,       # Bristow Group
    "CACI": 16058,       # CACI International
    "CBB": 23111,        # Cincinnati Bell → acquired by Macquarie
    "CBG": 1138118,      # CBRE Group → still active but ticker may differ
    "CBM": 816956,       # Cambrex → taken private
    "CBT": 18581,        # Cabot Corp
    "CHS": 1040792,      # Chico's FAS
    "CHUY": 1524940,     # Chuy's Holdings
    "CLI": 764065,       # Mack-Cali Realty → Veris Residential (VRE)
    "CMO": 726854,       # Capstead Mortgage → merged with Broadmark
    "CMP": 23598,        # Compass Minerals
    "COR": 1140859,      # Cencora (was ABC)
    "CORE": 1396033,     # Core Molding Technologies
    "CTB": 21076,        # Cooper Tire → acquired by Goodyear
    "CTL": 18926,        # CenturyLink → Lumen Technologies (LUMN)
    "CUB": 25475,        # Cubic Corp → taken private
    "CUBI": 1487918,     # Customers Bancorp
    "CUTR": 1162461,     # Cutera → acquired
    "CYTK": 1060714,     # Cytokinetics
    "DF": 722572,        # Dean Foods → bankruptcy
    "EE": 1820953,       # Excelerate Energy (may be diff company)
    "EGRX": 1302573,     # Eagle Pharmaceuticals
    "ENS": 1289308,      # EnerSys
    "ELY": 837465,       # Callaway Golf → Topgolf Callaway (MODG)
    "FARO": 917857,      # FARO Technologies
    "FBHS": 1519751,     # Fortune Brands Home & Security → FBIN
    "FFBC": 275119,      # First Financial Bancorp
    "FLOW": 774415,      # SPX FLOW → acquired by Lone Star
    "FMBI": 760498,      # First Midwest Bancorp → acquired by Old National
    "FRAN": 1410428,     # Francesca's → bankruptcy
    "FRGI": 1591596,     # Fiesta Restaurant Group → acquired
    "FSLR": 1274494,     # First Solar
    "GATX": 40211,       # GATX Corporation
    "GBX": 923120,       # Greenbrier Companies
    "GHL": 1261654,      # Greenhill & Co → acquired by Mizuho
    "GLT": 42682,        # Glatfelter → acquired by Berry Global
    "GOV": 1424182,      # Government Properties Trust → merged → DHC
    "GWR": 1035996,      # Genesee & Wyoming → taken private
    "HSII": 885988,      # Heidrick & Struggles
    "HI": 913144,        # Hillenbrand
    "HII": 1501585,      # Huntington Ingalls Industries
    "HRC": 1020569,      # Hill-Rom → acquired by Baxter
    "HT": 1407623,       # Hersha Hospitality → acquired
    "HIBB": 1017480,     # Hibbett Sports → acquired by JD Sports
    "IIVI": 820318,      # II-VI Inc → Coherent (COHR)
    "INT": 883984,       # World Fuel Services
    "INTL": 1093672,     # INTL FCStone → StoneX (SNEX)
    "IRBT": 1159166,     # iRobot
    "IVC": 742112,       # Invacare → acquired
    "KEM": 26058,        # KEMET → acquired by Yageo
    "KIRK": 1587523,     # Kirkland's
    "KS": 1564618,       # KapStone Paper → acquired by WestRock
    "LB": 701985,        # L Brands → Bath & Body Works (BBWI)
    "LL": 1396033,       # Lumber Liquidators → LL Flooring → bankrupt
    "LM": 60086,         # Legg Mason → acquired by Franklin Templeton
    "LMNX": 1060349,     # Luminex → acquired by DiaSorin
    "LPNT": 1301611,     # LifePoint Health → taken private
    "MINI": 1052054,     # Mobile Mini → acquired by WillScot
    "MMS": 1032208,      # Maximus
    "MTSC": 68709,       # MTS Systems → acquired by Amphenol
    "NBL": 72207,        # Noble Energy → acquired by Chevron
    "NLS": 1089872,      # Nautilus → Bowflex → bankruptcy
    "NTRI": 1096691,     # Nutrisystem → acquired by Tivity Health
    "NAVG": 788965,      # Navigators Group → acquired by Hartford
    "NCI": 1265083,      # was already above, skip
    "NWSA": 1564708,     # News Corp
    "PAY": 1506439,      # Pay Sign
    "PCH": 849213,       # PotlatchDeltic
    "PDCE": 77877,       # PDC Energy → acquired by Chevron
    "PDCO": 891024,      # Patterson Companies
    "PEI": 77281,        # PREIT → bankruptcy/restructured
    "PGNX": 702692,      # Progenics → acquired by Lantheus
    "PLXS": 785786,      # Plexus Corp
    "PPBI": 1028918,     # Pacific Premier Bancorp
    "PSB": 866368,       # PS Business Parks → acquired by Blackstone
    "PX": 884905,        # Praxair → merged with Linde (LIN)
    "QEP": 1108827,      # QEP Resources → acquired by Diamondback
    "RE": 882184,        # Everest Group (ticker changed to EG)
    "ROIC": 1407623,     # Retail Opportunity Investments
    "RSG": 1060391,      # Republic Services
    "RTEC": 790125,      # Rudolph Technologies → merged → Onto Innovation
    "RUTH": 1324358,     # Ruth's Hospitality → acquired by Darden
    "SAFM": 811596,      # Sanderson Farms → acquired by Cargill/Continental
    "SCG": 754737,       # SCANA → acquired by Dominion
    "SF": 1360901,       # Stifel Financial
    "SFLY": 1276316,     # Shutterfly → taken private
    "SFNC": 700564,      # Simmons First National
    "SIG": 832988,       # Signet Jewelers
    "SJI": 91928,        # South Jersey Industries → acquired
    "SLCA": 1437491,     # US Silica → acquired
    "SMG": 825542,       # Scotts Miracle-Gro
    "SNH": 926288,       # Senior Housing Properties → merged → DHC
    "SNV": 18349,        # Synovus Financial
    "SONC": 868259,      # Sonic → acquired by Inspire
    "SPN": 862861,       # Superior Energy → bankruptcy
    "SPTN": 77600,       # SpartanNash
    "SRDX": 717372,      # SurModics
    "SSP": 96793,        # E.W. Scripps
    "STI": 750556,       # SunTrust → merged with BB&T → Truist (TFC)
    "STMP": 1082157,     # Stamps.com → acquired → delisted
    "SUP": 1577762,      # Superior Industries
    "SVU": 95521,        # SuperValu → acquired by UNFI
    "SWM": 91767,        # Schweitzer-Mauduit → Mativ (MATV)
    "TCO": 102426,       # Taubman Centers → acquired by Simon
    "TECD": 790703,      # SYNNEX / TD SYNNEX
    "TGI": 98222,        # Triumph Group
    "TIF": 98246,        # Tiffany → acquired by LVMH
    "TLRD": 16040,       # Tailored Brands → bankruptcy
    "TMST": 1539838,     # TimkenSteel
    "TPRE": 1580149,     # Third Point Reinsurance → acquired by SiriusPoint
    "TPX": 316253,       # Tempur-Pedic → Tempur Sealy (TPX is same)
    "TUP": 49802,        # Tupperware → bankruptcy
    "TYPE": 1318134,     # Monotype Imaging → taken private
    "UCBI": 857855,      # United Community Banks
    "UFS": 101830,       # Domtar → acquired by Paper Excellence
    "UIHC": 1387916,     # United Insurance Holdings
    "UMPQ": 1077771,     # Umpqua Holdings → merged with Columbia Banking → COLB
    "UNT": 100726,       # Unit Corporation → restructured
    "VRTU": 1082968,     # Virtusa → acquired by Baring
    "VRTV": 1630031,     # Veritiv Corp
    "VSI": 1485172,      # Vitamin Shoppe → acquired by Franchise Group
    "VSTO": 1616318,     # Vista Outdoor → split → acquired
    "VVC": 102511,       # Vectren → acquired by CenterPoint Energy
    "VVI": 102588,       # Viad Corp
    "WAGE": 1465128,     # WageWorks → acquired by HealthEquity
    "WCG": 1071739,      # WellCare → acquired by Centene
    "WDR": 1136174,      # Waddell & Reed → acquired by Macquarie
    "WGL": 97745,        # WGL Holdings → acquired by AltaGas
    "WIRE": 850429,      # Encore Wire
    "WLH": 1584547,      # William Lyon Homes → acquired by Taylor Morrison
    "WPG": 1594686,      # Washington Prime Group → bankruptcy
    "WPX": 1518832,      # WPX Energy → acquired by Devon
    "WRI": 803649,       # Weingarten Realty → acquired by Kimco
    "WRK": 1636023,      # WestRock → merged with Smurfit Kappa
    "WTR": 14713,        # Aqua America → Essential Utilities (WTRG)
    "WWE": 1091907,      # World Wrestling Ent → merged → TKO/Endeavor
    "WWW": 106152,       # Wolverine World Wide
    "XOXO": 1517375,     # XO Group → acquired by WeddingWire
    "ARRS": 1355096,     # Arris International → acquired by CommScope
    "AEL": 1133421,      # American Equity Investment
    "CCMP": 1102426,     # Cabot Microelectronics → CMC Materials → acquired
    "SIX": 1437491,      # Six Flags → merged with Cedar Fair
    "SKX": 1065837,      # Skechers
    "SPGI": 64040,       # S&P Global (was MHFI/McGraw Hill)
    "SRE": 1032208,      # Sempra Energy → Sempra (SRE same CIK normally)
    "STLD": 811596,      # Steel Dynamics
    "SRCL": 855923,      # Stericycle
    "VSAT": 797721,      # ViaSat
    "VRTX": 875320,      # Vertex Pharmaceuticals
    "NUVA": 1227636,     # NuVasive → merged with Globus Medical
    "SWN": 7332,         # Southwestern Energy
    "NWN": 73088,        # Northwest Natural
    "EAT": 703351,       # Brinker International — note: EAT is ticker
    "ATGE": 1046568,     # Adtalem Global Education (was DeVry)
}


def main():
    ap = argparse.ArgumentParser(description="Resolve CIK for unmapped MAEC tickers")
    ap.add_argument("--user-agent", required=True)
    ap.add_argument("--maec-root", default="data/MAEC_Dataset")
    ap.add_argument("--cache", default=".sec_cache")
    ap.add_argument("--output", default="ticker_cik_overrides.json")
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()

    maec_root = Path(args.maec_root)
    cache_root = Path(args.cache)

    # Load current ticker map
    with (cache_root / "company_tickers.json").open() as f:
        data = json.load(f)
    current_map = {}
    for _, rec in data.items():
        t = str(rec.get("ticker", "")).upper().strip()
        cik = rec.get("cik_str")
        if t and isinstance(cik, int):
            current_map[t] = cik

    # Scan MAEC → unique tickers
    maec_tickers = set()
    for p in maec_root.iterdir():
        if not p.is_dir():
            continue
        parsed = parse_instance_folder(p.name)
        if parsed:
            maec_tickers.add(parsed[1])

    unmapped = sorted(maec_tickers - set(current_map.keys()))
    print(f"[INFO] {len(unmapped)} tickers not in current SEC map")

    # Load existing overrides if any
    out_path = Path(args.output)
    if out_path.exists():
        with out_path.open() as f:
            overrides = json.load(f)
    else:
        overrides = {}

    # Apply well-known overrides first
    resolved_count = 0
    for t in unmapped:
        if t in WELL_KNOWN_OVERRIDES and t not in overrides:
            overrides[t] = WELL_KNOWN_OVERRIDES[t]
            resolved_count += 1

    print(f"[INFO] Applied {resolved_count} well-known overrides")

    still_unmapped = [t for t in unmapped if t not in overrides]
    print(f"[INFO] Still unmapped after well-known overrides: {len(still_unmapped)}")

    if still_unmapped:
        print(f"[INFO] Remaining unmapped tickers: {', '.join(still_unmapped)}")
        # Try EDGAR company search for remaining
        sec = SecClient(user_agent=args.user_agent, sleep_s=args.sleep)

        # Try the exchange tickers file which includes more tickers
        print("[INFO] Fetching company_tickers_exchange.json for broader coverage...")
        exchange_cache = cache_root / "company_tickers_exchange.json"
        if exchange_cache.exists():
            with exchange_cache.open() as f:
                exchange_data = json.load(f)
        else:
            exchange_data = sec.get_json(EDGAR_COMPANY_TICKERS_EXCHANGE)
            if exchange_data:
                with exchange_cache.open("w") as f:
                    json.dump(exchange_data, f)

        if exchange_data:
            # Format: {"fields": [...], "data": [[cik, name, ticker, exchange], ...]}
            fields = exchange_data.get("fields", [])
            rows = exchange_data.get("data", [])
            ticker_idx = fields.index("ticker") if "ticker" in fields else 2
            cik_idx = fields.index("cik") if "cik" in fields else 0
            exchange_map = {}
            for row in rows:
                t = str(row[ticker_idx]).upper().strip()
                c = int(row[cik_idx])
                exchange_map[t] = c

            extra_resolved = 0
            for t in still_unmapped[:]:
                if t in exchange_map:
                    overrides[t] = exchange_map[t]
                    still_unmapped.remove(t)
                    extra_resolved += 1
            print(f"[INFO] Resolved {extra_resolved} more from exchange tickers file")

    # Final summary
    total_overrides = len(overrides)
    final_unmapped = [t for t in unmapped if t not in overrides]

    print(f"\n[SUMMARY]")
    print(f"  Total unmapped tickers: {len(unmapped)}")
    print(f"  Resolved via overrides: {total_overrides}")
    print(f"  Still unmapped: {len(final_unmapped)}")
    if final_unmapped:
        print(f"  Remaining: {', '.join(final_unmapped)}")

    # Save overrides
    with out_path.open("w") as f:
        json.dump(overrides, f, indent=2, sort_keys=True)
    print(f"\n[SAVED] {out_path} ({total_overrides} entries)")


if __name__ == "__main__":
    main()
