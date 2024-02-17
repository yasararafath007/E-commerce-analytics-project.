import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
file_path = 'encoded_classification_data.csv'
df = pd.read_csv(file_path)

# Suppress Streamlit warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page title
st.title("Exploratory Data Analysis (EDA) App")

# Display the first few rows of the dataset
st.subheader("Dataset Overview")
st.write(df.head())

def generate_heatmap():
    st.subheader("Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    st.pyplot(fig)

generate_heatmap()

def generate_pie_chart():
    st.subheader("Target Column Distribution (Pie Chart)")
    target_distribution = df['has_converted'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(target_distribution, labels=target_distribution.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

generate_pie_chart()

def generate_distribution_plot():
    st.subheader("Distribution of Feature Variables")
    selected_feature = st.selectbox("Select a feature for distribution plot", df.columns[:-1])
    distribution_plot = sns.histplot(data=df, x=selected_feature, hue='has_converted', kde=True)
    st.pyplot(distribution_plot.figure)

generate_distribution_plot()

def generate_relationship_plot():
    st.subheader("Relationship Plot")
    x_variable = st.selectbox("Select X-axis variable", df.columns[:-1])
    y_variable = st.selectbox("Select Y-axis variable", df.columns[:-1])
    relationship_plot = sns.relplot(data=df, x=x_variable, y=y_variable, hue='has_converted', palette='viridis')
    st.pyplot(relationship_plot)

generate_relationship_plot()
