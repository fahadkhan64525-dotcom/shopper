import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Product catalog
products = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053",  "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART", 3.39),
    ("22752",  "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("21730",  "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22633",  "HAND WARMER UNION JACK", 1.85),
    ("22632",  "HAND WARMER RED POLKA DOT", 1.85),
    ("21212",  "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("23343",  "JUMBO BAG RED RETROSPOT", 1.65),
    ("23166",  "MEDIUM CERAMIC TOP STORAGE JAR", 1.25),
    ("22993",  "SET OF 4 PANTRY JELLY MOULDS", 1.25),
    ("22086",  "PAPER CHAIN KIT 50'S CHRISTMAS", 2.55),
    ("85099B", "JUMBO BAG PINK POLKADOT", 2.08),
    ("22197",  "POPCORN HOLDER", 0.85),
    ("22196",  "LARGE POPCORN HOLDER", 1.25),
    ("21977",  "PACK OF 60 PINK PAISLEY CAKE CASES", 0.55),
    ("21212",  "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("22423",  "REGENCY CAKESTAND 3 TIER", 12.75),
    ("47566",  "PARTY BUNTING", 4.95),
    ("20727",  "LUNCH BAG BLACK SKULL", 1.65),
    ("22720",  "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("22721",  "WOOD STAMP SET FAIRY TALE", 1.25),
    ("21931",  "JUMBO STORAGE BAG SUKI", 2.08),
    ("22698",  "PINK REGENCY TEACUP AND SAUCER", 1.25),
    ("22699",  "ROSES REGENCY TEACUP AND SAUCER", 1.25),
    ("22700",  "BLUE REGENCY TEACUP AND SAUCER", 1.25),
    ("21928",  "JUMBO BAG WOODLAND ANIMALS", 2.08),
    ("21929",  "JUMBO BAG BAROQUE BLACK WHITE", 2.08),
    ("22382",  "LUNCH BOX I LOVE LONDON", 1.45),
    ("22383",  "LUNCH BAG SUKI DESIGN", 1.65),
    ("22384",  "LUNCH BAG PINK POLKADOT", 1.65),
    ("22385",  "LUNCH BAG CARS BLUE", 1.65),
    ("22386",  "LUNCH BAG WOODLAND", 1.65),
    ("23170",  "LARGE GLASS STORAGE JAR", 1.95),
    ("22961",  "JAM MAKING SET WITH JARS", 4.25),
    ("22960",  "JAM MAKING SET PRINTED", 4.25),
    ("22411",  "JUMBO SHOPPER VINTAGE RED PAISLEY", 2.08),
    ("22412",  "JUMBO BAG VINTAGE DOILY", 2.08),
]

countries = ['United Kingdom', 'Germany', 'France', 'EIRE', 'Spain',
             'Netherlands', 'Belgium', 'Switzerland', 'Portugal', 'Australia',
             'Norway', 'Italy', 'Channel Islands', 'Finland', 'Cyprus']

country_weights = [0.85, 0.03, 0.03, 0.02, 0.01,
                   0.01, 0.01, 0.01, 0.005, 0.005,
                   0.005, 0.005, 0.003, 0.003, 0.002]

# Generate 25000 rows
n_customers = 800
customer_ids = [f"C{10000 + i}" for i in range(n_customers)]
start_date = datetime(2022, 1, 1)
end_date   = datetime(2023, 12, 31)

rows = []
invoice_counter = 500000

for _ in range(25000):
    customer = random.choice(customer_ids)
    country  = random.choices(countries, weights=country_weights)[0]
    n_items  = random.randint(1, 8)
    invoice  = f"{invoice_counter}"
    invoice_counter += 1
    days_offset = random.randint(0, (end_date - start_date).days)
    inv_date = start_date + timedelta(days=days_offset)

    for _ in range(n_items):
        prod = random.choice(products)
        qty  = random.randint(1, 12)
        rows.append({
            "InvoiceNo":   invoice,
            "StockCode":   prod[0],
            "Description": prod[1],
            "Quantity":    qty,
            "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice":   prod[2],
            "CustomerID":  customer,
            "Country":     country,
        })

# Inject some NaN CustomerIDs (~5%)
df = pd.DataFrame(rows)
null_idx = df.sample(frac=0.05).index
df.loc[null_idx, "CustomerID"] = np.nan

# Inject some cancelled invoices (~3%)
cancel_idx = df.sample(frac=0.03).index
df.loc[cancel_idx, "InvoiceNo"] = "C" + df.loc[cancel_idx, "InvoiceNo"]
df.loc[cancel_idx, "Quantity"]  = -df.loc[cancel_idx, "Quantity"]

df.to_csv("/home/claude/ecommerce_data.csv", index=False)
print(f"Dataset shape: {df.shape}")
print(df.head(3))
