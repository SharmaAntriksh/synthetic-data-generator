# # sales_logic.py — optimized (Arrow-first, vectorized, memory-conscious)
# # Compatible replacement for your existing logic. Returns pyarrow.Table when possible.
# import numpy as np
# import pandas as pd
# import pyarrow as pa

# PA_AVAILABLE = pa is not None

# # ============================================================
# # GLOBAL WORKER STATE (injected via bind_globals)
# # ============================================================
# _G_skip_order_cols = None
# _G_product_np = None
# _G_customers = None
# _G_date_pool = None
# _G_date_prob = None
# _G_store_keys = None
# _G_promo_keys_all = None
# _G_promo_pct_all = None
# _G_promo_start_all = None
# _G_promo_end_all = None
# _G_store_to_geo_arr = None
# _G_geo_to_currency_arr = None
# _G_store_to_geo = None
# _G_geo_to_currency = None
# _G_file_format = None

# _fmt = lambda dt: np.char.replace(np.datetime_as_string(dt, unit='D'), "-", "")

# def bind_globals(gdict):
#     """Multiprocessing worker initializer inserts globals."""
#     globals().update(gdict)


# # ============================================================
# # CHUNK BUILDER
# # ============================================================
# def _build_chunk_table(n, seed, no_discount_key=1):
#     """
#     Build n synthetic sales rows.
#     Returns a pyarrow.Table when pyarrow is available.
#     """
#     rng = np.random.default_rng(seed)
#     skip_cols = _G_skip_order_cols

#     # ---- references ----
#     product_np = _G_product_np
#     customers = _G_customers
#     date_pool = _G_date_pool
#     date_prob = _G_date_prob
#     store_keys = _G_store_keys
#     promo_keys_all = _G_promo_keys_all
#     promo_pct_all = _G_promo_pct_all
#     promo_start_all = _G_promo_start_all
#     promo_end_all = _G_promo_end_all

#     st2g_arr = _G_store_to_geo_arr
#     g2c_arr = _G_geo_to_currency_arr
#     store_to_geo = _G_store_to_geo
#     geo_to_currency = _G_geo_to_currency

#     # cached lengths to avoid repeated len(...) calls
#     _len_date_pool = len(date_pool)
#     _len_customers = len(customers)
#     _len_store_keys = len(store_keys)
#     _len_products = len(product_np)

#     # ---------------------------------------------------------
#     # PRODUCTS
#     # ---------------------------------------------------------
#     prod_idx = rng.integers(0, _len_products, size=n)
#     prods = product_np[prod_idx]  # shape (n, cols)

#     # ensure numeric dtypes once (copy=False avoids copying when unnecessary)
#     product_keys = prods[:, 0]
#     unit_price    = prods[:, 1].astype(np.float64, copy=False)
#     unit_cost     = prods[:, 2].astype(np.float64, copy=False)

#     # STORE -> GEO -> CURRENCY (vectorized fast path, faster safe indexing)
#     store_key_arr = store_keys[rng.integers(0, len(store_keys), size=n)].astype(np.int64)

#     if st2g_arr is not None and g2c_arr is not None:
#         # fast path only if arrays are 1-D and sized appropriately
#         try:
#             max_key = int(store_key_arr.max()) if store_key_arr.size else -1
#             if st2g_arr.ndim == 1 and g2c_arr.ndim == 1 and max_key < st2g_arr.shape[0]:
#                 # use direct indexing (fast, vectorized)
#                 geo_arr = st2g_arr[store_key_arr]
#                 currency_arr = g2c_arr[geo_arr].astype(np.int64, copy=False)
#             else:
#                 raise IndexError("mapping arrays too small")
#         except Exception:
#             # fallback safe path using dicts
#             geo_arr = np.fromiter((store_to_geo.get(int(s), 0) for s in store_key_arr), dtype=np.int64, count=n)
#             currency_arr = np.fromiter((geo_to_currency.get(int(g), 0) for g in geo_arr), dtype=np.int64, count=n)
#     else:
#         geo_arr = np.fromiter((store_to_geo.get(int(s), 0) for s in store_key_arr), dtype=np.int64, count=n)
#         currency_arr = np.fromiter((geo_to_currency.get(int(g), 0) for g in geo_arr), dtype=np.int64, count=n)

#     # ---------------------------------------------------------
#     # QUANTITY
#     # ---------------------------------------------------------
#     qty = np.clip(rng.poisson(3, n) + 1, 1, 10)

#     # ---------------------------------------------------------
#     # ORDER GROUPING (vectorized generation of order-level data)
#     # ---------------------------------------------------------
#     avg_lines = 2.0
#     order_count = max(1, int(n / avg_lines))

#     # suffix & order dates (vectorized)
#     suffix_int = rng.integers(0, 999999, order_count, dtype=np.int64)
#     suffix = np.char.zfill(suffix_int.astype(str), 6)

#     od_idx = rng.choice(len(date_pool), size=order_count, p=date_prob)
#     order_dates = date_pool[od_idx]  # numpy datetime64 array

#     # vectorized date string (only create the string ids if we need them)
#     date_str = np.datetime_as_string(order_dates, unit='D')  # 'YYYY-MM-DD'
#     date_str = np.char.replace(date_str, "-", "")  # 'YYYYMMDD'

#     # ORDER IDS (integer form always needed for grouping)
#     # Create integer order id directly (YYYYMMDD + suffix) without building full python strings
#     # Build suffix numeric and combine to integer
#     date_int = date_str.astype(np.int64)  # YYYYMMDD as int
#     order_ids_int = (date_int * 1_000_000) + suffix_int  # integer order id

#     # Only create human-readable zero-padded string IDs when order columns are requested
#     if not skip_cols:
#         # create suffix string and padded string only when needed
#         suffix = np.char.zfill(suffix_int.astype(str), 6)
#         order_ids_str = np.char.add(date_str, suffix)  # numpy char array
#     else:
#         order_ids_str = None

#     # customers
#     cust_idx = rng.integers(0, len(customers), order_count)
#     order_customers = customers[cust_idx].astype(np.int64)

#     # lines per order
#     lines_per_order = rng.choice([1, 2, 3, 4, 5], order_count, p=[0.55, 0.25, 0.10, 0.06, 0.04])
#     expanded_len = lines_per_order.sum()
#     order_idx = np.repeat(np.arange(order_count), lines_per_order)

#     starts = np.repeat(np.cumsum(lines_per_order) - lines_per_order, lines_per_order)
#     line_num = (np.arange(expanded_len) - starts + 1).astype(np.int64)

#     # expand order-level arrays to line-level
#     # expand order-level arrays to line-level
#     # integer order id is always needed for grouping
#     sales_order_num_int = np.repeat(order_ids_int, lines_per_order)

#     # sales_order_num (string) only when requested
#     if order_ids_str is None:
#         sales_order_num = None
#     else:
#         sales_order_num = np.repeat(order_ids_str, lines_per_order)

#     customer_keys = np.repeat(order_customers, lines_per_order).astype(np.int64)
#     order_dates_expanded = np.repeat(order_dates, lines_per_order)

#     # pad if needed to reach exactly n rows
#     # use integer id length as truth (always present)
#     curr_len = len(sales_order_num_int)
#     if curr_len < n:
#         extra = n - curr_len

#         # choose extra dates and suffixes
#         ext_dates = date_pool[rng.choice(_len_date_pool, size=extra, p=date_prob)]
#         ext_suffix_int = rng.integers(0, 999999, extra, dtype=np.int64)

#         # numeric ext ids (fast, avoids string->int conversions)
#         ext_dt_str = _fmt(ext_dates)
#         ext_dt_int = ext_dt_str.astype(np.int64, copy=False)
#         ext_ids_int = (ext_dt_int * 1_000_000) + ext_suffix_int

#         if not skip_cols:
#             # build string ext ids only when needed (creates suffix strings from the same ints)
#             ext_suf = np.char.zfill(ext_suffix_int.astype(str), 6)
#             ext_ids_str = np.char.add(ext_dt_str, ext_suf)
#             sales_order_num = np.concatenate([sales_order_num, ext_ids_str]) if sales_order_num is not None else ext_ids_str

#         # append integer ids and extend other arrays
#         sales_order_num_int = np.concatenate([sales_order_num_int, ext_ids_int])
#         line_num = np.concatenate([line_num, np.ones(extra, dtype=np.int64)])
#         customer_keys = np.concatenate([customer_keys, customers[rng.integers(0, _len_customers, extra)]])
#         order_dates_expanded = np.concatenate([order_dates_expanded, ext_dates])



#     # trim/truncate exactly to n rows
#     sales_order_num_int = sales_order_num_int[:n]

#     # only trim string version if it exists
#     if sales_order_num is not None:
#         sales_order_num = sales_order_num[:n]

#     line_num = line_num[:n]
#     customer_keys = customer_keys[:n]
#     order_dates_expanded = order_dates_expanded[:n]


#     od_np = order_dates_expanded.astype("datetime64[D]")

#     # ---------------------------------------------------------
#     # DELIVERY / DUE DATE LOGIC (vectorized)
#     # ---------------------------------------------------------
#     hash_vals = sales_order_num_int  # integer grouping key

#     due_offset = (hash_vals % 5).astype(np.int64) + 3
#     due_date_np = od_np + due_offset.astype("timedelta64[D]")

#     line_seed = (product_keys + (hash_vals % 100)) % 100
#     order_seed = hash_vals % 100
#     product_seed = (hash_vals + product_keys) % 100


#     base_offset = np.zeros(n, dtype=np.int64)
#     mask_c = (60 <= order_seed) & (order_seed < 85) & (product_seed >= 60)
#     if mask_c.any():
#         base_offset[mask_c] = (line_seed[mask_c] % 4) + 1
#     mask_d = order_seed >= 85
#     if mask_d.any():
#         base_offset[mask_d] = (product_seed[mask_d] % 5) + 2

#     # early deliveries
#     early_mask = rng.random(n) < 0.10
#     early_days = rng.integers(1, 3, n)
#     delivery_offset = base_offset.copy()
#     delivery_offset[early_mask] = -early_days[early_mask]
#     delivery_date_np = due_date_np + delivery_offset.astype("timedelta64[D]")

#     # delivery_status as fixed-length numpy unicode array (avoids python object arrays)
#     delivery_status = np.full(n, "On Time", dtype='U15')
#     delivery_status[delivery_date_np < due_date_np] = "Early Delivery"
#     delivery_status[delivery_date_np > due_date_np] = "Delayed"

#     # ---------------------------------------------------------
#     # PROMOTIONS — vectorized for small promo lists, safe loop for large lists
#     # ---------------------------------------------------------
#     promo_keys = np.full(n, no_discount_key, dtype=np.int64)
#     promo_pct  = np.zeros(n, dtype=np.float64)

#     if promo_keys_all is not None and promo_keys_all.size > 0:

#         od_expanded = od_np[:, None]
#         start_ok = od_expanded >= promo_start_all
#         end_ok   = od_expanded <= promo_end_all
#         mask_all = start_ok & end_ok

#         rng = np.random.default_rng(seed)

#         NO_DISCOUNT_KEY = 1  # PromotionKey for no-discount promo

#         for i in range(n):
#             active = np.where(mask_all[i])[0]

#             if active.size == 0:
#                 # no promo for this date
#                 continue
            
#             # Filter out the No-Discount promo from active results
#             actual_promos = [
#                 j for j in active if promo_keys_all[j] != NO_DISCOUNT_KEY
#             ]

#             if actual_promos:
#                 idx = rng.choice(actual_promos)
#                 promo_keys[i] = promo_keys_all[idx]
#                 promo_pct[i]  = promo_pct_all[idx]
#             else:
#                 # ONLY the No-Discount promo is active → keep default value (1)
#                 promo_keys[i] = NO_DISCOUNT_KEY
#                 promo_pct[i]  = 0.0



#     # ---------------------------------------------------------
#     # DISCOUNTS
#     # ---------------------------------------------------------
#     promo_disc = unit_price * (promo_pct / 100.0)

#     rnd_pct = rng.choice([0, 5, 10, 15, 20], n, p=[0.85, 0.06, 0.04, 0.03, 0.02])
#     rnd_disc = unit_price * (rnd_pct * 0.01)

#     discount_amt = np.maximum(promo_disc, rnd_disc)
#     discount_amt *= rng.choice([0.90, 0.95, 1.00, 1.05, 1.10], n)
#     # round to quarters (0.25)
#     discount_amt = np.round(discount_amt * 4) / 4
#     discount_amt = np.minimum(discount_amt, unit_price - 0.01)

#     # ---------------------------------------------------------
#     # ORDER DELAY FLAG (order-level → line-level) using int grouping
#     # ---------------------------------------------------------
#     delayed_line = (delivery_status == "Delayed")
#     _, inv_idx = np.unique(sales_order_num_int, return_inverse=True)
#     delayed_any = np.bincount(inv_idx, weights=delayed_line).astype(bool)
#     is_order_delayed = delayed_any[inv_idx].astype(np.int8)


#     # ---------------------------------------------------------
#     # FINAL PRICE & COST TRANSFORM
#     # ---------------------------------------------------------
#     factor = rng.uniform(0.43, 0.61, size=n)
#     final_unit_price = np.round(unit_price * factor, 2)
#     final_unit_cost = np.round(unit_cost * factor, 2)
#     final_discount_amt = np.round(discount_amt * factor, 2)
#     final_net_price = np.round(final_unit_price - final_discount_amt, 2)
#     final_net_price = np.clip(final_net_price, 0.01, None)

#     # ---------------------------------------------------------
#     # OUTPUT: PYARROW OR PANDAS (Arrow-first, minimizing copies)
#     # ---------------------------------------------------------
#     if PA_AVAILABLE:
#         # --- Precast dates once (your current block recasts twice) ---
#         # these are already datetime64[D] views in our pipeline; avoid re-casting (copy-free)
#         od_d = od_np
#         due_d = due_date_np
#         del_d = delivery_date_np


#         # --- Build Arrow arrays (no schema changes, same columns) ---
#         cols = {}

#         # 1. Order columns (only if SkipOrderCols = False)
#         if not skip_cols:
#             cols["SalesOrderNumber"] = pa.array(sales_order_num, pa.string())
#             cols["SalesOrderLineNumber"] = pa.array(line_num, pa.int64())

#         # 2. Core dimension / key columns
#         cols["CustomerKey"] = pa.array(customer_keys, pa.int64())
#         cols["ProductKey"] = pa.array(product_keys, pa.int64())
#         cols["StoreKey"] = pa.array(store_key_arr, pa.int64())
#         cols["PromotionKey"] = pa.array(promo_keys, pa.int64())
#         cols["CurrencyKey"] = pa.array(currency_arr, pa.int64())

#         # 3. Dates
#         cols["OrderDate"]    = pa.array(od_d)
#         cols["DueDate"]      = pa.array(due_d)
#         cols["DeliveryDate"] = pa.array(del_d)

#         # 4. Measures
#         cols["Quantity"]        = pa.array(qty, pa.int64())
#         cols["NetPrice"]        = pa.array(final_net_price, pa.float64())
#         cols["UnitCost"]        = pa.array(final_unit_cost, pa.float64())
#         cols["UnitPrice"]       = pa.array(final_unit_price, pa.float64())
#         cols["DiscountAmount"]  = pa.array(final_discount_amt, pa.float64())

#         # 5. Status columns
#         cols["DeliveryStatus"]  = pa.array(delivery_status, pa.string())
#         cols["IsOrderDelayed"]  = pa.array(is_order_delayed, pa.int8())

        
#         # --- Partition columns (Arrow) ---
#         # compute year/month from od_d (datetime64[D]) in a vectorized, copy-free way
#         # convert to months-since-1970 then derive year and month
#         months_since_1970 = od_d.astype("datetime64[M]").astype("int64")
#         year_arr = (months_since_1970 // 12 + 1970).astype("int16")
#         month_arr = (months_since_1970 % 12 + 1).astype("int8")

#         if _G_file_format == "deltaparquet":
#             cols["Year"] = pa.array(year_arr, pa.int16())
#             cols["Month"] = pa.array(month_arr, pa.int8())

#         return pa.table(cols)

#     else:
#         df = {}

#         if not skip_cols:
#             df["SalesOrderNumber"]     = sales_order_num.astype(str)
#             df["SalesOrderLineNumber"] = line_num

#         df["CustomerKey"] = customer_keys
#         df["ProductKey"]  = product_keys
#         df["StoreKey"]    = store_key_arr
#         df["PromotionKey"] = promo_keys
#         df["CurrencyKey"]  = currency_arr

#         df["OrderDate"]    = od_np
#         df["DueDate"]      = due_date_np.astype("datetime64[D]")
#         df["DeliveryDate"] = delivery_date_np.astype("datetime64[D]")

#         df["Quantity"]       = qty
#         df["NetPrice"]       = final_net_price
#         df["UnitCost"]       = final_unit_cost
#         df["UnitPrice"]      = final_unit_price
#         df["DiscountAmount"] = final_discount_amt

#         df["DeliveryStatus"] = delivery_status
#         df["IsOrderDelayed"] = is_order_delayed

#         # --- Partition columns (NumPy/Pandas) ---
#         months_since_1970 = od_np.astype("datetime64[M]").astype("int64")
#         year_np = (months_since_1970 // 12 + 1970).astype("int16")
#         month_np = (months_since_1970 % 12 + 1).astype("int8")

#         if _G_file_format == "deltaparquet":
#             df["Year"] = year_np
#             df["Month"] = month_np

#         return pd.DataFrame(df)
