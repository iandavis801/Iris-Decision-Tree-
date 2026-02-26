import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import joblib
import io
import streamlit.components.v1 as components

# ==========================================
# 1. Page & Basic Configuration
# ==========================================
st.set_page_config(page_title="Iris Decision Tree Deep Dive", page_icon="🌳", layout="wide")
st.title("🌳 Iris Decision Tree Deep Dive & Visualization System")
st.markdown("""
This is an interactive dashboard designed to showcase the **Decision Tree** workflow.
You can switch between different analysis stages using the tabs above, dynamically adjust hyperparameters in the left panel, and instantly observe changes in the model's logic.
""")

# ==========================================
# 2. Load Data (Cache data to improve performance)
# ==========================================
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.data, iris.target, iris.target_names, iris.feature_names

df, X, y, target_names, feature_names = load_data()

# ==========================================
# 3. Sidebar - Parameter Control & Export
# ==========================================
st.sidebar.header("⚙️ Decision Tree Hyperparameters")

# Detail control parameters for Decision Tree
max_depth = st.sidebar.slider("Max Depth (max_depth)", 1, 10, 3, help="Controls the maximum depth of the tree. A value too high may lead to overfitting.")
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, help="The minimum number of samples required to split an internal node.")
criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"), help="The function to measure the quality of a split.")
test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, step=0.05)

# Build and Train Model
model = DecisionTreeClassifier(
    max_depth=max_depth, 
    min_samples_split=min_samples_split, 
    criterion=criterion, 
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
model.fit(X_train, y_train)

# Sidebar: Model Export Functionality
st.sidebar.divider()
st.sidebar.header("💾 Model Deployment & Export")
st.sidebar.markdown("Save the currently trained model parameters as a `.pkl` file.")
model_buffer = io.BytesIO()
joblib.dump(model, model_buffer)
model_bytes = model_buffer.getvalue()

st.sidebar.download_button(
    label="⬇️ Download Model (.pkl)",
    data=model_bytes,
    file_name="decision_tree_iris.pkl",
    mime="application/octet-stream"
)

# ==========================================
# 4. Main Layout: Tabs
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Exploratory Data Analysis (EDA)", 
    "🧠 2. Model Evaluation & Features", 
    "🌳 3. Decision Tree Logic",
    "🚀 4. Load Model & Live Prediction"
])

# ------------------------------------------
# Tab 1: EDA
# ------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(15), width='stretch')
        
    with col2:
        st.subheader("Interactive Feature Distribution (2D/3D)")
        
        # Let user dynamically select 2 to 3 features
        selected_features = st.multiselect(
            "Select features to observe (2-3 only):",
            options=feature_names,
            default=feature_names[:3], # Default to first three (3D)
            max_selections=3
        )
        
        # Dynamically draw 2D or 3D plot based on selection
        if len(selected_features) == 3:
            fig_dist = px.scatter_3d(
                df, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                color='species', opacity=0.8, size_max=10
            )
            fig_dist.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_dist, width='stretch')
            
        elif len(selected_features) == 2:
            fig_dist = px.scatter(
                df, x=selected_features[0], y=selected_features[1],
                color='species'
            )
            fig_dist.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_dist, width='stretch')
            
        else:
            # If selected less than 2, give a friendly warning
            st.warning("⚠️ Please select exactly 2 or 3 features to render a 2D or 3D plot!")

    st.divider()
    st.subheader("Feature Scatter Matrix")
    st.markdown("By increasing the chart height, you can clearly observe the clustering distributions and boundary relationships between different features.")
    fig_matrix = px.scatter_matrix(
        df, dimensions=feature_names, color="species",
        height=800  # <--- Optimized height to fix crowded charts
    )
    fig_matrix.update_traces(diagonal_visible=False) # Hide diagonal to make the plot cleaner
    fig_matrix.update_layout(margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig_matrix, width='stretch')

# ------------------------------------------
# Tab 2: Model Evaluation & Feature Analysis
# ------------------------------------------
with tab2:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display core metrics
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Training Samples", len(X_train))
    col_metric2.metric("Testing Samples", len(X_test))
    col_metric3.metric("Overall Accuracy", f"{accuracy * 100:.2f}%")
    
    st.divider()
    
    # Feature Importance & Confusion Matrix
    col_eval1, col_eval2 = st.columns([1.2, 1]) # Set column width ratio
    
    with col_eval1:
        st.subheader("🌟 Feature Importance")
        st.markdown("The decision tree algorithm evaluates the contribution weight of each feature during classification.")
        # Draw feature importance bar chart
        importances = model.feature_importances_
        fig_importance = px.bar(
            x=importances, y=feature_names, orientation='h',
            labels={'x': 'Importance Weight (0~1)', 'y': 'Feature'},
            color=importances, color_continuous_scale='Blues'
        )
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importance, width='stretch')
        
    with col_eval2:
        st.subheader("🎯 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, 
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=target_names, y=target_names,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cm, width='stretch')

    # Classification Report
    st.subheader("📄 Detailed Classification Report")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, width='stretch')

# ------------------------------------------
# Tab 3: Decision Tree Logic Visualization
# ------------------------------------------
with tab3:
    st.subheader("🌳 Internal Decision Logic (White-box Model)")
    st.markdown("""
    **High-Res Scrollable Window:** You can adjust `max_depth` in the sidebar to observe how the tree is pruned.
    The SVG vector graphic below supports lossless zooming and dragging, perfectly solving the node overlapping issue.
    """)
    
    depth = model.get_depth()
    n_leaves = model.get_n_leaves()
    
    # Dynamic canvas size design
    dynamic_width = max(12, n_leaves * 2)
    dynamic_height = max(8, depth * 2.5)
    
    fig_tree, ax_tree = plt.subplots(figsize=(dynamic_width, dynamic_height))
    
    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,      
        rounded=True,     
        fontsize=12,
        ax=ax_tree
    )
    
    # SVG rendering and HTML container
    svg_buffer = io.StringIO()
    fig_tree.savefig(svg_buffer, format="svg", bbox_inches='tight')
    svg_code = svg_buffer.getvalue()
    plt.close(fig_tree)
    
    html_string = f"""
    <div style="width: 100%; height: 600px; overflow: auto; border: 1px solid #ddd; border-radius: 5px; padding: 10px; background-color: white;">
        {svg_code}
    </div>
    """
    components.html(html_string, height=620)

# ------------------------------------------
# Tab 4: Model Load & Live Prediction (Deployment Demo)
# ------------------------------------------
with tab4:
    st.subheader("🚀 Simulated Production: Load & Predict")
    st.markdown("""
    This section simulates a real-world production environment. You can upload the `.pkl` model file downloaded from the sidebar,
    and input custom flower features to test if the model loads correctly and performs inference.
    """)
    
    # Create file upload section
    uploaded_file = st.file_uploader("📂 Upload the trained model file (.pkl)", type="pkl")
    
    if uploaded_file is not None:
        try:
            # Read and restore model
            loaded_model = joblib.load(uploaded_file)
            st.success("✅ Successfully restored the model from the file! You can now make predictions.")
            
            st.divider()
            st.subheader("🧪 Input Features for Prediction")
            
            # Use four columns for user to input feature values (default to common Setosa values)
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                sepal_l = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
            with col_f2:
                sepal_w = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
            with col_f3:
                petal_l = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            with col_f4:
                petal_w = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
            
            # Predict button
            if st.button("🔮 Predict", type="primary"):
                # Assemble input into sklearn's required 2D array format
                input_data = [[sepal_l, sepal_w, petal_l, petal_w]]
                
                # Perform prediction
                prediction_idx = loaded_model.predict(input_data)[0]
                predicted_species = target_names[prediction_idx]
                
                # Get prediction probability (If depth is enough, usually 100% one class; if restricted, there will be ratios)
                probs = loaded_model.predict_proba(input_data)[0]
                
                # Display highlighting results
                st.markdown(f"### 🎉 The model predicts this Iris as: **<span style='color:blue'>{predicted_species}</span>**", unsafe_allow_html=True)
                
                # Draw probability distribution chart
                prob_df = pd.DataFrame({"Species": target_names, "Probability": probs})
                fig_prob = px.bar(
                    prob_df, x="Probability", y="Species", orientation='h', 
                    color="Species", title="Prediction Probability Distribution",
                    text_auto='.0%'
                )
                fig_prob.update_layout(xaxis_range=[0, 1])
                st.plotly_chart(fig_prob, width='stretch')
                
        except Exception as e:
            st.error(f"❌ Failed to load file. Ensure it is a valid scikit-learn model file. Error: {e}")
    else:
        st.info("👆 First, click the download button in the left sidebar to get the model, then drag the `.pkl` file into the dashed box above.")
