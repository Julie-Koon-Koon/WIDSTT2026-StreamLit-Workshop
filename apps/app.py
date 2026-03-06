"""
Titanic Workshop Dashboard

Single Streamlit app with 3 tabs, built from the artifacts created by
notebooks/workshop.py.  The code here mirrors the notebook structure so
you can copy-paste helper functions and variable names between them.

Tab 1 "Explore the Data" - interactive EDA (mirrors Q1 & Q2)
Tab 2 "Predict & Explain" - model predictions (mirrors Q4)
Tab 3 "Key Takeaways" - stakeholder summary (mirrors Q5)

Run with:  streamlit run apps/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Paths  (same pattern as the notebook - relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "artifacts", "titanic_clean_basic.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "artifacts", "model.joblib")

# Page config
st.set_page_config(page_title="Titanic Workshop Dashboard", layout="wide")


# HELPER: Load data & model  (cached so they only run once)
@st.cache_data
def load_data():
    """Load the clean CSV produced by notebooks/workshop.py (Q3 output)."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_model():
    """Load the trained pipeline produced by notebooks/workshop.py (Q4 output)."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# HELPER: Apply sidebar filters  (used in Tab 1)
def apply_filters(df):
    """Sidebar multiselects for Gender, Class, Embarked. Returns filtered df."""
    st.sidebar.header("Filters")

    sex_filter = st.sidebar.multiselect(
        "Gender",
        options=sorted(df["Sex"].unique()),
        default=sorted(df["Sex"].unique()),
    )
    class_filter = st.sidebar.multiselect(
        "Ticket Class",
        options=sorted(df["Pclass"].unique()),
        default=sorted(df["Pclass"].unique()),
    )
    embarked_opts = sorted(df["Embarked"].dropna().unique())
    embarked_filter = st.sidebar.multiselect(
        "Port of Embarkation",
        options=embarked_opts,
        default=embarked_opts,
    )

    mask = (
        df["Sex"].isin(sex_filter)
        & df["Pclass"].isin(class_filter)
        & df["Embarked"].isin(embarked_filter)
    )
    return df[mask]


# HELPER: Build input DataFrame for prediction  (mirrors notebook Q4)
# Same column order as the clean CSV (minus Survived).
def build_input_row(pclass, sex, age, sibsp, parch, fare, embarked, title):
    """Create a single-row DataFrame the model pipeline can consume."""
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    return pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked],
        "Title": [title],
        "FamilySize": [family_size],
        "IsAlone": [is_alone],
    })


# HELPER: SHAP waterfall for one passenger  (mirrors notebook Q4 SHAP)
def get_shap_explanation(model, input_df):
    """Return a matplotlib figure with the SHAP waterfall plot, or None."""
    if not SHAP_AVAILABLE:
        return None

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    X_transformed = preprocessor.transform(input_df)

    # Feature names - same lists as the notebook
    numeric_features = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
    cat_features = ["Pclass", "Sex", "Embarked", "Title"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names = numeric_features + list(ohe.get_feature_names_out(cat_features))

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    # Handle both list-style and 3D-array-style returns (mirrors notebook fix)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_value = explainer.expected_value[1]
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values[0]
        base_value = explainer.expected_value

    fig, _ = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv,
            base_values=base_value,
            data=X_transformed[0],
            feature_names=feature_names,
        ),
        show=False,
    )
    return fig


# TAB 1 - Explore the Data  (interactive version of Q1 + Q2)
def tab_explore(df):
    st.header("Explore the Titanic Dataset")
    st.markdown("Use the **sidebar filters** to slice the data.")

    dff = apply_filters(df)

    #  Overview metrics 
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Passengers", len(dff))
    c2.metric("Survivors", int(dff["Survived"].sum()))
    rate = dff["Survived"].mean() * 100 if len(dff) > 0 else 0
    c3.metric("Survival Rate", f"{rate:.1f}%")

    #  4 key charts (same groupby logic as notebook Q2) 
    col_l, col_r = st.columns(2)

    with col_l:
        # Survival by Gender
        gender_data = dff.groupby("Sex")["Survived"].mean().reset_index()
        fig1 = px.bar(gender_data, x="Sex", y="Survived", color="Sex",
                       title="Survival Rate by Gender", range_y=[0, 1],
                       labels={"Survived": "Survival Rate"})
        st.plotly_chart(fig1, use_container_width=True)

        # Age distribution by survival
        fig3 = px.histogram(dff, x="Age", color="Survived", nbins=25,
                             title="Age Distribution by Survival",
                             color_discrete_map={0: "#EF553B", 1: "#00CC96"})
        st.plotly_chart(fig3, use_container_width=True)

    with col_r:
        # Survival by Class
        class_data = dff.groupby("Pclass")["Survived"].mean().reset_index()
        fig2 = px.bar(class_data, x="Pclass", y="Survived", color="Pclass",
                       title="Survival Rate by Class", range_y=[0, 1],
                       labels={"Survived": "Survival Rate"})
        st.plotly_chart(fig2, use_container_width=True)

        # Class + Gender interaction
        cg = dff.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()
        fig4 = px.bar(cg, x="Pclass", y="Survived", color="Sex",
                       barmode="group", title="Survival by Class & Gender",
                       range_y=[0, 1], labels={"Survived": "Survival Rate"})
        st.plotly_chart(fig4, use_container_width=True)

    #  Factor explorer (mirrors Q3 features) 
    st.subheader("Factor Explorer")
    factor_options = [c for c in ["FamilySize", "IsAlone", "Title"] if c in dff.columns]
    if factor_options:
        factor = st.selectbox("Choose a factor to explore", factor_options)
        fig_f = px.histogram(dff, x=factor, color="Survived", barmode="group",
                              title=f"Survival by {factor}",
                              color_discrete_map={0: "#EF553B", 1: "#00CC96"})
        st.plotly_chart(fig_f, use_container_width=True)

    #  Raw data viewer 
    with st.expander("View Raw Data"):
        st.dataframe(dff, use_container_width=True)
        st.caption(f"Showing {len(dff)} rows")


# TAB 2 - Predict & Explain  (interactive version of Q4)
def tab_predict(model):
    st.header("Predict & Explain Survival")

    if model is None:
        st.error("Model not found. Run `python notebooks/workshop.py` first.")
        return

    st.markdown("Enter passenger details, then click **Predict**.")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("Passenger Details")
        pclass   = st.selectbox("Ticket Class", [1, 2, 3], index=2)
        sex      = st.selectbox("Gender", ["male", "female"])
        age      = st.slider("Age", 0, 100, 30)
        sibsp    = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
        parch    = st.number_input("Parents / Children Aboard", 0, 10, 0)
        fare     = st.number_input("Fare (£)", 0.0, 520.0, 32.0, step=5.0)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        title    = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

        predict_btn = st.button("Predict Survival", type="primary",
                                use_container_width=True)

    # Build the input row (uses helper above)
    input_df = build_input_row(pclass, sex, age, sibsp, parch, fare, embarked, title)

    if predict_btn:
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        with col_result:
            st.subheader("Prediction Result")
            m1, m2 = st.columns(2)
            m1.metric("Survival Probability", f"{probability:.1%}")
            m2.metric("Prediction", "SURVIVED" if prediction == 1 else "NOT SURVIVED")

            if prediction == 1:
                st.success(f"**Survived** - {probability:.1%} probability")
            else:
                st.error(f"**Did Not Survive** - {probability:.1%} probability")

            # SHAP explanation (uses helper above)
            st.markdown("---")
            st.subheader("Why This Prediction?")
            fig_shap = get_shap_explanation(model, input_df)
            if fig_shap is not None:
                st.pyplot(fig_shap)
                plt.close(fig_shap)
                st.caption("**Red** = pushes toward survival; **blue** = pushes away.")
            elif not SHAP_AVAILABLE:
                st.info("Install `shap` to see feature explanations.")
    else:
        with col_result:
            st.subheader("Prediction Result")
            st.info("Fill in the details on the left and click **Predict Survival**.")

    # Try-it-yourself prompts
    st.markdown("---")
    st.subheader("Try It Yourself!")
    st.markdown("""
    1. **A wealthy woman** - 1st class, female, age 30, fare £80
    2. **A young boy** - 3rd class, male, age 5, fare £8
    3. **An older gentleman** - 1st class, male, age 60, fare £50
    4. **A large family** - 3rd class, 4 siblings

    *Change one feature at a time to see what the model cares about most.*
    """)


# TAB 3 - Key Takeaways  (stakeholder summary - mirrors Q5)
def tab_takeaways(df, model):
    st.header("Key Takeaways")
    st.markdown("A summary you could present to a non-technical stakeholder.")

    #  Top findings 
    st.subheader("1. Top Findings from EDA")
    st.markdown("""
    | Finding | Detail |
    |---------|--------|
    | **Overall survival** | Only ~38% of passengers survived |
    | **Gender gap** | Women survived at ~74% vs. men at ~19% |
    | **Class gap** | 1st class: ~63%, 2nd: ~47%, 3rd: ~24% |
    | **Age matters** | Children (< 10) had higher survival rates |
    | **Interaction** | Almost all 1st-class women survived; most 3rd-class men did not |
    """)

    #  Model performance 
    st.subheader("2. Model Performance")
    if model is not None:
        try:
            X = df.drop("Survived", axis=1)
            y = df["Survived"]
            acc = accuracy_score(y, model.predict(X))
            auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
            m1, m2 = st.columns(2)
            m1.metric("Accuracy (full dataset)", f"{acc:.1%}")
            m2.metric("ROC-AUC (full dataset)", f"{auc:.3f}")
            st.caption("Held-out test metrics are in the notebook output.")
        except Exception:
            st.info("Could not compute metrics - check model/data compatibility.")
    else:
        st.info("Model not loaded. Run the notebook first.")

    #  Recommendations 
    st.subheader("3. Recommendations")
    st.markdown("""
    **Domain perspective:**
    - Equal access to lifeboats - survival shouldn't depend on ticket class.
    - Family-friendly evacuation protocols for all group sizes.
    - Better record-keeping (77% of Cabin data was missing).

    **Data science perspective:**
    - A simple Random Forest with ~10 features achieves strong performance.
    - Top features (Sex, Pclass, Title) match domain knowledge - the model
      learned real patterns, not noise.
    - SHAP lets us verify the model's reasoning matches reality.
    """)

    #  Workflow recap 
    st.subheader("4. The Workflow We Followed")
    st.markdown("""
    ```
    Raw Data -> Q1: Understand -> Q2: Discover patterns
                                       ↓
    Dashboard ← Q4: Model & Explain ← Q3: Engineer features
    ```
    Each step built on the previous one.
    """)


# MAIN - wire everything together
def main():
    st.title("Titanic Workshop: From Data to Dashboard")

    df = load_data()
    model = load_model()

    if df is None:
        st.error("Data not found. Run `python notebooks/workshop.py` first.")
        return

    tab1, tab2, tab3 = st.tabs(
        ["Explore the Data", "Predict & Explain", "Key Takeaways"]
    )
    with tab1:
        tab_explore(df)
    with tab2:
        tab_predict(model)
    with tab3:
        tab_takeaways(df, model)


if __name__ == "__main__":
    main()
