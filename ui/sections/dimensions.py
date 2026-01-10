import streamlit as st

DIMENSION_SIZE_FIELDS = {
    "customers": "total_customers",
    "products": "num_products",
    "stores": "num_stores",
    "promotions": "num_seasonal",
}

def render_dimensions(cfg):
    st.subheader("4️⃣ Dimensions")

    def dim(section, label, step, min_val=1):
        field = DIMENSION_SIZE_FIELDS[section]
        cfg[section][field] = st.number_input(
            label,
            min_value=min_val,
            step=step,
            value=cfg[section][field],
        )

    dim("customers", "Customers (entities)", step=1_000)
    dim("products", "Products (SKUs)", step=500)
    dim("stores", "Physical stores", step=10)
    dim("promotions", "Active promotions", step=5, min_val=0)
