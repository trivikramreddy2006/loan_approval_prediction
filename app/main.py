import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time
import os

model_path = os.path.join("model", "loan_model.pkl")
with open(model_path, "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    encoders = saved_data["encoders"]
    scaler = saved_data["scaler"]
    metrics = saved_data["metrics"]

def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1

def preprocess_inputs(inputs):
    mapped = {
        "no_of_dependents": inputs["no_of_dependents"],
        "education": safe_transform(encoders["education"], inputs["education"]),
        "self_employed": safe_transform(encoders["self_employed"], inputs["self_employed"]),
        "income_annum": inputs["income_annum"],
        "loan_amount": inputs["loan_amount"],
        "loan_term": inputs["loan_term"],
        "cibil_score": inputs["cibil_score"],
        "residential_assets_value": inputs["residential_assets_value"],
        "commercial_assets_value": inputs["commercial_assets_value"],
        "luxury_assets_value": inputs["luxury_assets_value"],
        "bank_asset_value": inputs["bank_asset_value"],
    }
    arr = np.array(list(mapped.values())).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled

def make_prediction(inputs):
    x = preprocess_inputs(inputs)
    prediction = model.predict(x)[0]
    prob = model.predict_proba(x)[0][1] * 100
    return prediction, prob

def main():
    st.set_page_config(page_title="Loan Approval Prediction App", page_icon="💳", layout="wide")
    st.title("Loan Approval Prediction App")
    st.markdown("### Only for educational purposes. Not financial advice")

    tabs = st.tabs([
        "📋 Applicant Input",
        "🔮 Prediction & Insights",
        "📊 Data Explorer",
        "📈 Model Performance",
        "⚡ Loan Simulation"
    ])

    with tabs[0]:
        st.header("Applicant Input")

        col1, col2, col3 = st.columns(3)
        with col1:
            no_of_dependents = st.number_input("No. of Dependents", 0, 10, 0)
            education = st.selectbox("Education", encoders["education"].classes_)
            self_employed = st.selectbox("Self Employed", encoders["self_employed"].classes_)
            income_annum = st.number_input("Annual Income", 0, 100000000, 1000000, step=50000)
        with col2:
            loan_amount = st.number_input("Loan Amount", 0, 100000000, 500000, step=50000)
            loan_term = st.number_input("Loan Term (months)", 1, 360, 12)
            cibil_score = st.number_input("CIBIL Score", 300, 900, 750)
            residential_assets_value = st.number_input("Residential Assets Value", 0, 100000000, 0, step=50000)
        with col3:
            commercial_assets_value = st.number_input("Commercial Assets Value", 0, 100000000, 0, step=50000)
            luxury_assets_value = st.number_input("Luxury Assets Value", 0, 100000000, 0, step=50000)
            bank_asset_value = st.number_input("Bank Asset Value", 0, 100000000, 0, step=50000)

        if st.button("Submit Application"):
            st.session_state["inputs"] = {
                "no_of_dependents": no_of_dependents,
                "education": education,
                "self_employed": self_employed,
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets_value,
                "commercial_assets_value": commercial_assets_value,
                "luxury_assets_value": luxury_assets_value,
                "bank_asset_value": bank_asset_value,
            }
            placeholder=st.empty()
            placeholder.success("Application submitted. Go to Prediction tab.")
            time.sleep(5)
            placeholder.empty()

    with tabs[1]:
        st.header("Prediction & Insights")

        if "inputs" not in st.session_state:
            st.warning("Please submit applicant details first.")
        else:
            st.subheader("Applicant Summary")
            st.json(st.session_state["inputs"])

            prediction, prob = make_prediction(st.session_state["inputs"])
            if prediction == 1:
                monthly_payment = st.session_state["inputs"]["loan_amount"] / st.session_state["inputs"]["loan_term"]
                st.write(f"📌 Monthly Payment (no interest): {monthly_payment:,.2f}")

                st.success(f"Loan Approved (Probability: {prob:.2f}%)")
            else:
                monthly_payment = st.session_state["inputs"]["loan_amount"] / st.session_state["inputs"]["loan_term"]
                st.write(f"📌 Monthly Payment (no interest): {monthly_payment:,.2f}")
                st.error(f"Loan Rejected (Probability: {prob:.2f}%)")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={"text": "Approval Probability"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            
            st.plotly_chart(fig, use_container_width=True)

       # Data Explorer Tab
    with tabs[2]:
        st.header("📊 Data Explorer")

        # Load dataset
        df = pd.read_csv(
            r"D:\DJANGO_COURSE_2.xx (2)\DJANGO_COURSE_2.xx\python_learning\loan_prediction_app\data\loan_approval_dataset.csv"
        )
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20))

        st.subheader("Filters")
        col1, col2 = st.columns(2)

        with col1:
            edu_filter = st.selectbox(
                "Filter by Education", ["All"] + df["education"].unique().tolist(), key="edu_filter"
            )
            emp_filter = st.selectbox(
                "Filter by Self Employed", ["All"] + df["self_employed"].unique().tolist(), key="emp_filter"
            )

        with col2:
            min_cibil, max_cibil = st.slider(
                "CIBIL Score Range", 300, 900, (300, 900), key="cibil_filter"
            )
            status_filter = st.selectbox(
                "Loan Status", ["All"] + df["loan_status"].unique().tolist(), key="status_filter"
            )

        # Apply filters
        filtered_df = df.copy()
        if edu_filter != "All":
            filtered_df = filtered_df[filtered_df["education"] == edu_filter]
        if emp_filter != "All":
            filtered_df = filtered_df[filtered_df["self_employed"] == emp_filter]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["loan_status"] == status_filter]
        filtered_df = filtered_df[
            (filtered_df["cibil_score"] >= min_cibil) & (filtered_df["cibil_score"] <= max_cibil)
        ]

        st.subheader("Filtered Data")
        st.dataframe(filtered_df)

        # Visualizations
        st.subheader("Visualizations")

        if not filtered_df.empty:
            fig1 = px.histogram(
                filtered_df, x="cibil_score", color="loan_status", barmode="overlay",
                title="CIBIL Score Distribution by Loan Status"
            )
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.scatter(
                filtered_df, x="income_annum", y="loan_amount",
                color="loan_status", size="cibil_score",
                hover_data=["education", "self_employed"],
                title="Income vs Loan Amount (Colored by Status, Sized by CIBIL)"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning(" No data available for selected filters.")

    with tabs[3]:
        st.header("📈 Model Performance")
        st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        st.metric("Precision", f"{metrics['precision']:.2f}")
        st.metric("Recall", f"{metrics['recall']:.2f}")
        st.metric("F1 Score", f"{metrics['f1']:.2f}")

    
    # Loan Simulation Tab
    with tabs[4]:
        st.header("⚡ Loan Simulation")
        st.markdown("Adjust applicant features to see how prediction changes.")

        if "inputs" not in st.session_state:
            st.warning(" Please submit applicant details first in the Applicant Input tab.")
        else:
            sim_inputs = st.session_state["inputs"].copy()

            st.subheader("Adjust Key Features")

            col1, col2 = st.columns(2)
            with col1:
                new_income = st.slider(
                    "Applicant Income (Annual)", 
                    0, 100000000, sim_inputs["income_annum"], step=50000
                )
                new_loan = st.slider(
                    "Loan Amount", 
                    0, 100000000, sim_inputs["loan_amount"], step=50000
                )
                new_term = st.slider(
                    "Loan Term (months)", 
                    1, 360, sim_inputs["loan_term"], step=1
                )
            with col2:
                new_cibil = st.slider(
                    "CIBIL Score", 
                    300, 900, sim_inputs["cibil_score"], step=10
                )
                new_education = st.selectbox(
                    "Education", encoders["education"].classes_,
                    index=list(encoders["education"].classes_).index(sim_inputs["education"]),
                    key="edu_sim"
                )   

                new_self_emp = st.selectbox(
                    "Self Employed", encoders["self_employed"].classes_,
                    index=list(encoders["self_employed"].classes_).index(sim_inputs["self_employed"]),
                    key="selfemp_sim"
                )

           
            sim_inputs["income_annum"] = new_income
            sim_inputs["loan_amount"] = new_loan
            sim_inputs["loan_term"] = new_term
            sim_inputs["cibil_score"] = new_cibil
            sim_inputs["education"] = new_education
            sim_inputs["self_employed"] = new_self_emp

            sim_pred, sim_prob = make_prediction(sim_inputs)

            st.subheader("Simulated Prediction Result")
            if sim_pred == 1:
                st.success(f" Loan would be approved with probability {sim_prob:.2f}%")
            else:
                st.error(f" Loan would not be approved (Approval probability: {sim_prob:.2f}%)")

            fig_sim = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_prob,
                title={"text": "Simulated Approval Probability"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig_sim, use_container_width=True)

if __name__ == "__main__":
        main()
