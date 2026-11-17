import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)

API_URL = f"{API_BASE_URL}/predict"
BATCH_API_URL = f"{API_BASE_URL}/predict/batch"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None

st.sidebar.title("Customer Churn Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Single Customer Prediction",
        "Batch Customer Prediction",
        "Model Information"
    ]
)

st.title("Customer Churn Intelligence")
st.subheader("Customer Retention Prediction System")


if page == "Single Customer Prediction":
    with st.form("customer_input_form"):

        st.header("Customer Information")

        # 1. Customer Profile
        st.subheader("Customer Profile")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"]
            )

        with col2:
            partner = st.selectbox(
                "Partner",
                ["No", "Yes"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"]
            )

        st.divider()

        # 2. Account Information
        st.subheader("Account Information")

        col1, col2 = st.columns(2)

        with col1:
            tenure_months = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["No", "Yes"]
            )

        with col2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check"
                ]
            )

        st.divider()

        # 3. Services
        st.subheader("Services")

        col1, col2 = st.columns(2)

        with col1:
            phone_service = st.selectbox(
                "Phone Service",
                ["No", "Yes"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )

        with col2:
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

        st.divider()

        # 4. Charges
        st.subheader("Charges")

        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=0.5
            )

        with col2:
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            width='stretch'
        )

    if submitted:

        customer_data = {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,

            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,

            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,

            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method
        }

        try:
            response = requests.post(
                API_URL,
                json=customer_data,
                timeout=10
            )

            response.raise_for_status()

            prediction = response.json()

            st.success("Prediction generated successfully!")

            st.metric(
                label="Churn Probability",
                value=f"{prediction['churn_probability']:.2%}"
            )

            st.progress(
                prediction["churn_probability"],
                text="Churn Probability"
            )

            probability = prediction["churn_probability"]

            if probability < 0.30:
                st.success("Low Risk: Customer has a low estimated churn probability.")

            elif probability < 0.70:
                st.warning("Moderate Risk: Customer requires attention.")

            else:
                st.error("High Risk: Customer has a high estimated churn probability.")

        except requests.exceptions.HTTPError:
            try:
                error_detail = response.json().get("detail", "Invalid request")
            except ValueError:
                error_detail = response.text

            st.error(f"API Error: {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the prediction API. Check whether FastAPI is running.")

        except requests.exceptions.Timeout:
            st.error("The prediction request timed out.")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")




if page == "Batch Customer Prediction":

    st.header("Batch Customer Prediction")

    st.markdown(
        """
        Upload a CSV file containing multiple customer records
        to generate churn predictions in bulk.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        customers_df = pd.read_csv(uploaded_file)

        expected_columns = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "gender",
            "senior_citizen",
            "partner",
            "dependents",
            "phone_service",
            "multiple_lines",
            "internet_service",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
            "contract",
            "paperless_billing",
            "payment_method"
        ]

        missing_columns = set(expected_columns) - set(customers_df.columns)
        extra_columns = set(customers_df.columns) - set(expected_columns)

        if missing_columns or extra_columns:
            st.error("Invalid CSV structure.")

            if missing_columns:
                st.write("Missing columns:", sorted(missing_columns))

            if extra_columns:
                st.write("Unexpected columns:", sorted(extra_columns))

            st.stop()

        st.subheader("Uploaded Customer Data")

        st.write(f"Total customers uploaded: {len(customers_df)}")

        st.dataframe(
            customers_df,
            width='stretch'
        )

        st.divider()


        if st.button("Predict Churn for All Customers"):

            if len(customers_df) > 100:
                st.error("Maximum batch size allowed is 100 customers.")

            else:
                try:
                    response = requests.post(
                        BATCH_API_URL,
                        json={
                            "customers": customers_df.to_dict(
                                orient="records"
                            )
                        },
                        timeout=30
                    )

                    response.raise_for_status()

                    batch_result = response.json()

                    results_df = customers_df.copy()

                    predictions_df = pd.DataFrame(
                        batch_result["predictions"]
                    )

                    results_df["churn_probability"] = predictions_df[
                        "churn_probability"
                    ]

                    results_df["churn_prediction"] = predictions_df[
                        "churn_prediction"
                    ]

                    results_df["risk_category"] = results_df[
                        "churn_probability"
                    ].apply(
                        lambda x: (
                            "Low Risk" if x < 0.30
                            else "Moderate Risk" if x < 0.70
                            else "High Risk"
                        )
                    )

                    # Save results so they survive Streamlit reruns
                    st.session_state["batch_results"] = results_df

                    st.success(
                        f"Successfully generated predictions for "
                        f"{batch_result['total_customers']} customers!"
                    )

                except requests.exceptions.RequestException as e:
                    st.error(f"Batch prediction request failed: {e}")


    # Display saved results independently of the prediction button
    if st.session_state["batch_results"] is not None:

        results_df = st.session_state["batch_results"]

        # 1. KPI metrics
        total_customers = len(results_df)

        predicted_churners = (
            results_df["churn_prediction"] == 1
        ).sum()

        predicted_non_churners = (
            results_df["churn_prediction"] == 0
        ).sum()

        average_churn_probability = (
            results_df["churn_probability"].mean()
        )

        st.subheader("Batch Prediction Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Customers", total_customers)
        col2.metric("Predicted Churners", predicted_churners)
        col3.metric("Predicted Non-Churners", predicted_non_churners)
        col4.metric(
            "Average Churn Probability",
            f"{average_churn_probability:.2%}"
        )

        # 2. Risk distribution chart
        st.divider()

        st.subheader("Customer Risk Distribution")

        risk_order = ["Low Risk", "Moderate Risk", "High Risk"]

        risk_counts = (
            results_df["risk_category"]
            .value_counts()
            .reindex(risk_order, fill_value=0)
            .reset_index()
        )

        risk_counts.columns = ["Risk Category", "Number of Customers"]

        risk_colors = {
            "Low Risk": "#2ECC71",
            "Moderate Risk": "#F39C12",
            "High Risk": "#E74C3C"
        }

        fig = px.bar(
            risk_counts,
            x="Risk Category",
            y="Number of Customers",
            color="Risk Category",
            color_discrete_map=risk_colors,
            text="Number of Customers",
            category_orders={"Risk Category": risk_order}
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(size=15, color="white")
        )

        fig.update_layout(
            xaxis=dict(
                title=None,
                tickangle=0,
                tickfont=dict(size=16)
            ),
            yaxis=dict(
                title="Number of Customers",
                title_font=dict(size=14),
                tickfont=dict(size=13),
                rangemode="tozero"
            ),
            showlegend=False,
            height=450,
            margin=dict(t=35, b=55, l=45, r=25),
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font=dict(family="Arial", color="white")
        )

        st.plotly_chart(fig, width='stretch')

        # 3. High-risk customer filtering
        st.divider()

        st.subheader("High-Risk Customers")

        high_risk_customers = results_df[
            results_df["risk_category"] == "High Risk"
        ].copy()

        high_risk_customers = high_risk_customers.sort_values(
            by="churn_probability",
            ascending=False
        )

        minimum_probability = st.slider(
            "Minimum churn probability",
            min_value=0.70,
            max_value=1.00,
            value=0.70,
            step=0.01,
            format="%.2f"
        )

        filtered_customers = high_risk_customers[
            high_risk_customers["churn_probability"] >= minimum_probability
        ]

        high_risk_csv = filtered_customers.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download High-Risk Customers",
            data=high_risk_csv,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )

        st.write(
            f"Customers matching filter: {len(filtered_customers)}"
        )

        st.dataframe(
            filtered_customers[
                [
                    "tenure_months",
                    "contract",
                    "monthly_charges",
                    "churn_probability",
                    "risk_category"
                ]
            ],
            width='stretch'
        )

        # 4. Complete prediction results
        st.divider()

        st.subheader("Batch Prediction Results")

        st.dataframe(
            results_df,
            width='stretch'
        )

        csv_data = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Batch Prediction Results",
            data=csv_data,
            file_name="customer_churn_predictions.csv",
            mime="text/csv"
        )

if page == "Model Information":

    st.divider()

    st.subheader("About the Model")

    st.markdown(
        """
        The Customer Churn Prediction System uses a
        **Random Forest Classifier** to estimate the
        likelihood of a customer discontinuing their
        subscription.

        The model was developed using the IBM Telco
        Customer Churn dataset and trained on 19
        customer attributes.

        These attributes cover:

        - **Customer Profile:** Gender, senior citizen status,
          partner and dependent information.
        - **Account Information:** Tenure, contract type,
          billing method and payment method.
        - **Subscribed Services:** Internet, phone and
          additional service subscriptions.
        - **Financial Information:** Monthly charges and
          total charges.

        The model returns a churn probability between
        0 and 1, which can help identify customers who
        may require retention attention.
        """
    )

    st.divider()

    st.subheader("Model Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Algorithm",
        "Random Forest"
    )

    col2.metric(
        "Input Features",
        "19"
    )

    col3.metric(
        "Number of Trees",
        "424"
    )

    st.markdown(
        """
        The system uses a tuned Random Forest Classifier
        to estimate the probability of customer churn.

        The model was trained using 19 customer attributes,
        including account information, subscribed services,
        customer tenure, and billing details.
        """
    )

    st.divider()

    st.subheader("Model Hyperparameters")

    hyperparameters = pd.DataFrame({
        "Hyperparameter": [
            "n_estimators",
            "max_depth",
            "max_features",
            "min_samples_split",
            "min_samples_leaf",
            "class_weight"
        ],
        "Selected Value": [
            424,
            30,
            "log2",
            6,
            9,
            "balanced_subsample"
        ]
    })

    st.dataframe(
        hyperparameters,
        hide_index=True,
        width='stretch'
    )

    st.divider()

    st.subheader("Model Evaluation Metrics")

    st.caption(
        "Performance evaluated on the held-out test dataset."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "77.30%")
    col2.metric("Precision", "56.58%")
    col3.metric("Recall", "74.42%")

    col4, col5, col6 = st.columns(3)

    col4.metric("F1-score", "64.29%")
    col5.metric("ROC-AUC", "86.01%")
    col6.metric("Average Precision", "70.19%")

    st.divider()

    st.subheader("Confusion Matrix")

    st.caption(
        "Classification results on the held-out test dataset."
    )

    confusion_matrix = [
        [802, 221],
        [99, 288]
    ]

    cm_df = pd.DataFrame(
        confusion_matrix,
        index=["No Churn", "Churn"],
        columns=["No Churn", "Churn"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(
            x="Predicted Label",
            y="Actual Label",
            color="Customers"
        ),
        aspect="auto"
    )

    fig.update_traces(
        textfont=dict(
            size=25
        )
    )

    fig.update_layout(
        height=450,
        width=800,
        xaxis=dict(
            title_font=dict(size=25),
            tickfont=dict(size=15)
        ),
        yaxis=dict(
            title_font=dict(size=25),
            tickfont=dict(size=15),
            tickangle=-90,
            automargin=True
        )
    )

    st.plotly_chart(
        fig,
        width='content'
    )

    st.divider()

    st.subheader("Feature Importance")

    st.markdown(
        """
        The following features have the greatest influence
        on the model's predictive performance, measured
        using permutation importance on the test dataset.
        """
    )

    importance_df = pd.DataFrame({
        "Feature": [
            "Contract",
            "Dependents",
            "Tenure Months",
            "Internet Service",
            "Total Charges",
            "Tech Support",
            "Online Security",
            "Payment Method",
            "Monthly Charges",
            "Streaming TV"
        ],
        "Importance": [
            0.045252,
            0.028762,
            0.019411,
            0.017491,
            0.009179,
            0.004587,
            0.003245,
            0.003241,
            0.002583,
            0.001914
        ]
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Top 10 Features by Permutation Importance"
    )

    fig.update_traces(
        texttemplate="%{x:.4f}",
        textposition="outside",
        textfont=dict(size=13)
    )

    fig.update_layout(
        height=550,
        width=800,
        xaxis_title="Mean Decrease in ROC-AUC",
        yaxis_title=None,
        yaxis=dict(tickfont=dict(size=14)),
        xaxis=dict(
            title_font=dict(size=20),
            tickfont=dict(size=13)
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        width='content'
    )

    st.divider()

    st.subheader("Model Limitations")

    st.markdown(
        """
        Although the model demonstrates useful predictive
        performance, certain limitations should be considered
        when interpreting its predictions.

        - **Dataset dependency:** The model was trained on the
          IBM Telco Customer Churn dataset. Its performance may
          differ when applied to customers from other telecom
          providers or business environments.

        - **Probabilistic predictions:** Churn probabilities
          represent model estimates, not certainty about a
          customer's future behaviour.

        - **Correlation vs. Causation:** Feature importance
          indicates predictive relationships learned by the
          model. It does not establish that a particular
          customer attribute causes churn.

        - **Temporal limitations:** The model learns from
          historical observations. Changes in customer
          behaviour, pricing, services, or market conditions
          may affect its future performance.

        - **Business context:** A high predicted churn
          probability does not automatically mean that a
          retention intervention will be successful or
          economically beneficial.
        """
    )