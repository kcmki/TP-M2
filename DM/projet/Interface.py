import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

import streamlit as st
from dataset import Dataset, Dataset2
import plotly.express as px

import streamlit as st
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta 

data1 = pd.read_csv('Data/Dataset1.csv')
data2 = pd.read_csv('Data/Dataset2_correct.csv')



def dataset1():
    # Sidebar
    Set = Dataset(data1)
    Set.toNumeric()

    st.sidebar.header("Preprocessing Options")

    # Dropdowns for preprocessing options
    null_option = st.sidebar.selectbox("Handle Null Values", ["drop", "mean"])
    outliers_option = st.sidebar.selectbox("Handle Outliers", [None, "drop", "mean", "median", "Q1Q3"])
    normalization_option = st.sidebar.selectbox("Normalization", [None, "minmax", "zscore"])

    # Checkbox for discretization
    discretization_checkbox = st.sidebar.checkbox("Activate Discretization")

    ignore_list = st.sidebar.multiselect("Ignore List for Discretization", Set.data.columns)
    
    Set.resetData()
    Set.preprocessData(null=null_option, outliers=outliers_option,normalisation=normalization_option)
    if discretization_checkbox:
        Set.reduction(ignore=ignore_list)


    # Button to trigger preprocessing
    st.sidebar.markdown("---")
    if st.sidebar.button("Show data"):
        st.header("Visualisation des données")
        st.write("Taille des données ",Set.data.shape)
        st.write(Set.data)


    st.sidebar.markdown("---")
    st.sidebar.header("Plotting Options")


    # Button to display tendencies
    if st.sidebar.button("Show Tendencies"):
        st.header("Tendance centrale")
        Set.tendances()
        st.write(Set.tendance)

    # Button to display symmetries
    if st.sidebar.button("Show Symmetries"):
        st.header("Symetrie")
        Set.defSymetrie()
        st.write(Set.symetrie)


    

    if st.sidebar.button("box plots"):

        for box in Set.data.columns:
            st.header(f"Boxplot for {box}")

            # Clear the previous figure
            plt.clf()

            fig, ax = plt.subplots()
            ret = Set.data.boxplot(column=box, return_type='dict', ax=ax)
            ax.set_title(f"Boxplot for {box}")

            # Display fliers data
            fliers_data = pd.DataFrame(ret['fliers'][0].get_ydata()).transpose()
            st.write(fliers_data)

            # Display the plot
            st.pyplot(fig)

    if st.sidebar.button("correlation"):
        st.header("correlation")
        Set.setCorrelation()
        st.write(Set.correlation)
        pass

    if st.sidebar.button("scatter matrix"):
        st.header("scatter matrix")
        scatter_matrix(Set.data)
        st.pyplot()
        pass
    if st.sidebar.button("scatter plots"):
        st.header("scatter plot")
        for i in Set.data.columns:
            for j in Set.data.columns[Set.data.columns.get_loc(i)+1:]:
                plt.figure(figsize=(6, 4))
                plt.scatter(Set.data[i], Set.data[j])
                plt.xlabel(i)
                plt.ylabel(j)
                plt.title(f'Scatter plot of {i} and {j}')
                st.pyplot()


    if st.sidebar.button("histogram"):
        for col in Set.data.columns:
            st.header(f"Histogram for {col}")

            fig, ax = plt.subplots()
            Set.data.hist(column=col, ax=ax)
            ax.set_title(f"histogram for {col}")
            # Display the plot
            st.pyplot(fig)
            pass

def dataset2():

    st.sidebar.header("Choose what do you want to see")
    Set2 = Dataset2(data2)
    Set2.data["Start date"] = pd.to_datetime(Set2.data["Start date"])
    Set2.data["end date"] = pd.to_datetime(Set2.data["end date"])
    Set2.uniquesData("zcta")
    # Dropdowns for preprocessing options
    null_option = st.sidebar.selectbox("Handle Null Values", ["drop", "mean"])
    outliers_option = st.sidebar.selectbox("Handle Outliers", [None, "drop", "mean", "median", "Q1Q3"])
    normalization_option = st.sidebar.selectbox("Normalization", [None, "minmax", "zscore"])
    
    ignore_list = st.sidebar.multiselect("Ignore List for processing", Set2.data.columns,default=["Start date","end date","time_period","population","zcta"])
    Set2.preprocessData(null=null_option,outliers=outliers_option,normalisation=normalization_option,ignore=ignore_list)


    if st.sidebar.button("Show data"):
        st.header("Visualisation des données")
        st.write("Taille des données ",Set2.data.shape)
        st.write(Set2.data)

    st.sidebar.markdown("---")
    st.sidebar.header("Plotting Options")

    if st.sidebar.button("cas confirmés & tests positifs / zones"):
        Set2.barGraph(i="zcta",j="case count",title="case count by Zone")
        st.pyplot()
        Set2.barGraph(i="zcta",j="positive tests",title="positive tests by Zone")
        st.pyplot()
    st.sidebar.markdown("---")
    zoneDistributionAnnuel = st.sidebar.selectbox("Choix de zone", Set2.data["zcta"].unique())
    if st.sidebar.button("Distribution annuel / zone"):
        Set2.YearlyDistributionByZoneTests(zoneDistributionAnnuel)
        st.pyplot()
        Set2.YearlyDistributionByZonePTC(zoneDistributionAnnuel)
        st.pyplot()
    st.sidebar.markdown("---")
    if st.sidebar.button("distribution cas / zone ou année "):
        Set2.barYearCase()
        st.pyplot()
        Set2.barZoneCase()
        st.pyplot()
    st.sidebar.markdown("---")
    if st.sidebar.button("Nombre de test selon la population"):
        Set2.testByPopu()
        st.pyplot()
    st.sidebar.markdown("---")
    numberOfZones = st.sidebar.number_input("Chose a number of zones", step=1, value=1, format="%d",max_value=Set2.data["zcta"].nunique(),min_value=1)
    if st.sidebar.button("Nombre de cas selon la population"):
        Set2.mostAffectedZone(count=numberOfZones)
        st.pyplot()
    st.sidebar.markdown("---")
    start_time = st.sidebar.slider(
        "When do you start?",
        value=(Set2.data["Start date"].min().to_pydatetime(), Set2.data["end date"].max().to_pydatetime()), 
        format="DD/MM/YY"
    )
    zonedate = st.sidebar.selectbox("Choix de zone :", Set2.data["zcta"].unique())
    if st.sidebar.button("Cas confirmés & tests positifs & test realisé / zone & date"):
        X = pd.to_datetime(start_time[0])
        Y = pd.to_datetime(start_time[1])

        st.subheader(f"Zone : {zonedate} Date : {X.date()} -> {Y.date()}")

        Set2.CaseCountPosTestByZone(Zone=zonedate,start=X,end=Y)
        st.pyplot()
        Set2.posTestByTestCountByZone(Zone=zonedate,start=X,end=Y)
        st.pyplot()
    #selected_range = st.slider("Select a Range of Values", min_value, max_value, default_range)



def main():
    st.title("Data Preprocessing and Analysis")

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Select Dataset", ["Dataset1", "Dataset2"])

    if selected_page == "Dataset1":
        dataset1()
    else:
        dataset2()


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()


