
# Create a colourful and interactive student retention
# dashboard using the trained machine learning model.

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Configure the dashboard.

st.set_page_config(
    page_title="EduTrack AI",
    page_icon="🎓",
    layout="wide"
)

# Add custom colours and dashboard styling.

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg, #eef2ff, #fdf2f8, #ecfeff
    );
    color: #1e293b;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #312e81, #7c3aed
    );
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    color: white;
}

.hero {
    background: linear-gradient(
        120deg, #312e81, #7c3aed, #0891b2
    );
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 20px #00000020;
}

.hero h1 {
    color: white;
    font-size: 42px;
}

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border-top: 5px solid #8b5cf6;
    box-shadow: 0 4px 12px #00000015;
}

[data-testid="stMetric"] * {
    color: #312e81;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        90deg, #7c3aed, #2563eb
    );
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    width: 100%;
    padding: 12px;
}

[data-testid="stForm"] {
    background: #ffffff;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #ddd6fe;
}

h1, h2, h3 {
    color: #312e81;
}

</style>
""", unsafe_allow_html=True)

# Load the saved model and academic majors.

folder = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(folder / "dropout_model.pkl")

model = load_model()

majors = joblib.load(folder / "majors.pkl")

# Create the sidebar navigation.

st.sidebar.title("🎓 EduTrack AI")

st.sidebar.write("Student Retention Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Student Prediction",
        "📊 Model Performance",
        "ℹ️ About the Project"
    ]
)

st.sidebar.divider()

st.sidebar.write("CDS 4003")
st.sidebar.write("Data Science for Business Application")
st.sidebar.write("Fatima & Maitha")

# Create the dashboard header.

st.markdown("""
<div class="hero">
    <h1>🎓 EduTrack AI</h1>
    <p style="font-size:20px;">
    Student Retention Analytics & Early Warning System
    </p>
    <p>Understanding student dropout through machine learning.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational prototype only. "
    "Not suitable for real-world student decisions."
)

# PAGE 1: HOME DASHBOARD

if page == "🏠 Home":

    st.header("🏠 Dashboard Overview")

    st.write(
        "Explore student retention data, generate "
        "demonstration predictions and evaluate model performance."
    )

    # Display dataset summary cards.

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", "2,134")
    c2.metric("Continued", "1,773")
    c3.metric("Dropped Out", "361")
    c4.metric("Dropout Rate", "16.9%")

    st.divider()

    col1, col2 = st.columns(2)

    # Create a colourful student retention donut chart.

    with col1:

        st.subheader("Student Retention Distribution")

        fig = go.Figure(data=[go.Pie(
            labels=["Continued", "Dropped Out"],
            values=[1773, 361],
            hole=0.55,
            marker=dict(
                colors=["#10b981", "#f43f5e"]
            )
        )])

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#312e81"),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Display the project objectives.

    with col2:

        st.subheader("🎯 Project Objectives")

        st.info("🔍 Explore possible student dropout patterns.")

        st.success("📚 Support student retention research.")

        st.warning("📊 Compare machine learning models.")

        st.info("💡 Demonstrate an interactive prediction system.")

# PAGE 2: STUDENT PREDICTION

elif page == "🔮 Student Prediction":

    st.header("🔮 Student Risk Prediction")

    st.write(
        "Enter a hypothetical student profile "
        "to generate a demonstration prediction."
    )

    # Collect student information.

    with st.form("student_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("👤 Student Information")

            age = st.number_input("Age", 18, 29, 21)

            major = st.selectbox(
                "Academic Major", majors
            )

            gpa_current = st.slider(
                "Current GPA", 1.5, 4.0, 2.5
            )

            gpa_history = st.slider(
                "Historical GPA", 1.05, 3.99, 2.5
            )

            attendance = st.slider(
                "Attendance (%)", 50, 100, 75
            )

        with col2:

            st.subheader("📚 Academic Information")

            gap = st.number_input(
                "Enrollment Gap (Months)", 0, 3, 0
            )

            credits = st.number_input(
                "Credits Completed", 20, 119, 60
            )

            financial_aid = st.selectbox(
                "Financial Aid", ["No", "Yes"]
            )

            warnings = st.number_input(
                "Academic Warnings", 0, 3, 0
            )

            meetings = st.number_input(
                "Advisor Meetings", 0, 5, 2
            )

        submitted = st.form_submit_button(
            "✨ Analyse Student Risk"
        )

    # Prepare the student information for prediction.

    if submitted:

        student = pd.DataFrame([{
            "age": age,
            "major": major,
            "gpa_current": gpa_current,
            "gpa_history": gpa_history,
            "attendance_rate": attendance / 100,
            "enrollment_gap_months": gap,
            "credits_completed": credits,
            "financial_aid_status": int(
                financial_aid == "Yes"
            ),
            "academic_warning_count": warnings,
            "advisor_meeting_count": meetings,
            "gpa_change": gpa_current - gpa_history,
            "low_attendance": int(attendance < 75),
            "low_gpa": int(gpa_current < 2.0),
            "multiple_warnings": int(warnings >= 2)
        }])

        # Match the features used during training.

        student = student[model.feature_names_in_]

        # Generate the prediction and dropout score.

        prediction = model.predict(student)[0]

        probability = model.predict_proba(student)[0][1]

        score = probability * 100

        st.divider()
        st.header("📋 Prediction Results")

        col1, col2 = st.columns(2)

        # Create an interactive risk gauge.

        with col1:

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "%"},
                title={"text": "Model-estimated Dropout Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#312e81"},
                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#86efac"
                        },
                        {
                            "range": [30, 70],
                            "color": "#fde68a"
                        },
                        {
                            "range": [70, 100],
                            "color": "#fda4af"
                        }
                    ]
                }
            ))

            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#312e81"),
                height=350
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            st.caption(
                "The score is not a validated dropout "
                "probability. Gauge colours are illustrative."
            )

        # Display the predicted student status.

        with col2:

            st.subheader("Student Status")

            if prediction == 1:

                st.error("⚠️ Possible Student Dropout Risk")

            else:

                st.success("✅ Predicted to Continue Studying")

            st.metric("Current GPA", gpa_current)

            st.metric("Attendance", f"{attendance}%")

            st.metric("Academic Warnings", warnings)

        # Generate support suggestions from entered information.

        st.subheader("💡 Suggested Student Support")

        if gpa_current < 2.0:
            st.warning("📚 Consider tutoring and academic support.")

        if attendance < 75:
            st.warning("📅 Explore possible attendance difficulties.")

        if warnings >= 2:
            st.warning("👨‍🏫 Consider an academic advisor meeting.")

        if financial_aid == "No":
            st.info("💰 Discuss financial support options if needed.")

        if meetings == 0:
            st.info("🤝 Encourage communication with an academic advisor.")

        st.caption(
            "These suggestions are illustrative rules based "
            "on the entered information, not model explanations."
        )

# PAGE 3: MODEL PERFORMANCE

elif page == "📊 Model Performance":

    st.header("📊 Machine Learning Performance")

    st.success("Selected Model: Logistic Regression")

    # Display the actual model evaluation results.

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "51.29%")
    c2.metric("Recall", "54.17%")
    c3.metric("F1-score", "27.27%")
    c4.metric("ROC-AUC", "0.5108")

    # Create the model comparison dataset.

    performance = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Tuned Logistic Regression",
            "Tuned Random Forest"
        ],
        "Accuracy": [0.8314, 0.8314, 0.5129, 0.6956],
        "Precision": [0, 0, 0.1822, 0.1548],
        "Recall": [0, 0, 0.5417, 0.1806],
        "F1-score": [0, 0, 0.2727, 0.1667]
    })

    # Create an interactive model comparison chart.

    fig = px.bar(
        performance,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-score"],
        barmode="group",
        title="Model Performance Before and After Tuning",
        color_discrete_sequence=[
            "#38bdf8",
            "#f59e0b",
            "#10b981",
            "#8b5cf6"
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#312e81"),
        yaxis_range=[0, 1],
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(performance, use_container_width=True)

    # Display the actual confusion matrix.

    st.subheader("Confusion Matrix")

    matrix = px.imshow(
        [[180, 175], [33, 39]],
        labels=dict(
            x="Predicted Status",
            y="Actual Status",
            color="Students"
        ),
        x=["Continued", "Dropped Out"],
        y=["Continued", "Dropped Out"],
        text_auto=True,
        color_continuous_scale="Purples"
    )

    st.plotly_chart(matrix, use_container_width=True)

    st.warning(
        "The ROC-AUC score of 0.5108 indicates weak "
        "predictive performance. Further improvements "
        "are required before real-world implementation."
    )

# PAGE 4: ABOUT THE PROJECT

else:

    st.header("ℹ️ About the Project")

    st.subheader("Project Title")

    st.write(
        "Predicting University Student Dropout "
        "Using Machine Learning"
    )

    st.subheader("👩‍💻 Developed By")

    st.write("Fatima Salman & Maitha Hassan")

    st.subheader("🎓 Course")

    st.write("CDS 4003 – Data Science for Business Application")

    st.subheader("🤖 Machine Learning Models")

    st.write("Logistic Regression and Random Forest")

    st.subheader("🎯 Business Objective")

    st.write(
        "Explore student dropout prediction and demonstrate "
        "how machine learning could support retention research."
    )

    st.subheader("⚠️ Project Limitations")

    st.write(
        "The dataset is imbalanced and the selected model "
        "has limited predictive performance. This application "
        "is an educational demonstration and should not "
        "be used to make decisions about real students."
    )

# Create the dashboard footer.

st.divider()

st.markdown("""
<div style="text-align:center; color:#64748b; padding:15px;">
🎓 EduTrack AI | CDS 4003 | Fatima & Maitha
<br>
Student Retention Analytics Dashboard
</div>
""", unsafe_allow_html=True)
