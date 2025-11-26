# 📦 **contoso-fake-generator**
**A fully modular, configurable synthetic dataset generator inspired by Microsoft's Contoso Retail Data Warehouse.**

Generate high-quality **fact + dimension** datasets that mimic real retail behavior, with support for promotions, weighted customer behavior, store lifecycle, seasonality, and realistic delivery patterns.

Perfect for:
- Power BI demos  
- DAX training  
- SQL practice  
- Data modeling exercises  
- Benchmarking ETL tools  
- Building sample retail dashboards  

---

## 🚀 Features

### ✔ **Rich Dimension Generators**
- **Customers** with geo distribution + weighted behavior  
- **Stores** with open/close date windows  
- **Promotions** with timelines + discount logic  
- **Dates** table with fiscal calendar support  

### ✔ **Sales Fact Generator**
- Weighted dates (year growth, seasonality, weekday effects)
- Automatic no-sales days (5–10%)
- Promotion assignment based on active date ranges
- Realistic:
  - quantity distributions  
  - pricing + cost  
  - discount logic  
  - order → line expansion  
  - delivery delays / early delivery  

### ✔ **Chunk-based large file generation**
Generate **millions** of rows without running out of memory.

### ✔ **Merge or chunk output**
Optionally merge into a single `sales.parquet` file.

### ✔ **Config-driven pipeline**
Modify **config.json** to create customized dataset variants.

### ✔ **Automated output packaging**
Creates a final folder such as:

```
Customer 82K | Sales 600K | 2025-11-26_14-32-10/
```

And copies all generated parquet files inside it.

---

## 📁 Project Structure

```
contoso-fake-generator/
│ main.py
│ config.json
│ README.md
│
├─ src/
│   ├─ customers.py
│   ├─ stores.py
│   ├─ promotions.py
│   ├─ dates.py
│   ├─ sales.py
│   ├─ output_utils.py
│   └─ __init__.py
│
├─ data/
│   ├─ parquet_dims/
│   ├─ fact_out/
│   └─ Names/
│
└─ generated_datasets/
```

---

## 🔧 Installation

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Usage

```
python main.py
```

Output will appear inside:

```
generated_datasets/
```

---

## 🛠 Extending the Generator
Feel free to add:
- Products
- Inventory
- Returns
- Employees
- Territories
- More facts and dimensions

---

## 🧾 License
MIT License.
