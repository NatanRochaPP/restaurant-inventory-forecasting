# Data

`sales_data.csv` is a **synthetic** 24-month, single-site restaurant sales dataset
(14,620 rows; 20 SKUs; daily; 1 Jan 2024 – 31 Dec 2025). It contains no personal or
guest data.

It is generated deterministically by `../baseline_forecasting.py` (random seed = 42),
so it can be recreated at any time by running that script. Effect sizes (weekend uplift,
UK bank-holiday spikes, weather sensitivity, promotions, intermittent slow-movers) are
calibrated to magnitudes reported in the demand-forecasting literature.

### Columns
| column | description |
|---|---|
| `date` | calendar date |
| `sku` | product/ingredient name |
| `category` | Bakery / Produce / Meat / Dairy / Fish / Beverage / Ambient |
| `shelf_life_days` | perishability (days) |
| `units_sold` | daily units sold (target) |
| `temp_c` | synthetic daily mean temperature (°C) |
| `bank_holiday` | 1 if a UK (England & Wales) bank holiday |
| `school_holiday` | 1 during approximate England school holidays |
| `weekend` | 1 for Fri/Sat/Sun |
| `dow` | day of week (0 = Monday) |
| `promo` | 1 if the SKU was on promotion that day |
