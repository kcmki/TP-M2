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
from dataset3 import Apriori, calcul_confiance, calculate_intervals, dataset_to_discret, discretisation_par_amplitude_column, discretisation_par_taille_column, generate_association_rules, get_items

data1 = pd.read_csv('Data/Dataset1.csv')
data2 = pd.read_csv('Data/Dataset2_correct.csv')
data3 = pd.read_excel('Data/Dataset3up.xlsx')


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

def dataset3():
    global data3
    intervale_option = st.sidebar.selectbox("methode de discrétisation", ["Brooks Carruthers", "Huntsberger","Sturges"])
    disc_option = st.sidebar.selectbox("methode de discrétisation", ["fréquence egale", "largeur egale"])
    columns = st.sidebar.multiselect("choix des colonnes", data3.columns[:3])
    columns_bool = np.zeros(3) 
    for i in range(len(columns)):
        if (columns[i] == "Rainfall"):
            columns_bool[0] = 1
        if (columns[i] == "Humidity"):
            columns_bool[1] = 1
        if (columns[i] == "Temperature"):
            columns_bool[2] = 1
  

    discretiser = st.sidebar.checkbox("Discretiser")
    if discretiser:
        k = calculate_intervals(data3.shape[0],method=intervale_option)

        data3 = dataset_to_discret(data3,intervalles_type=disc_option, nombre_intervalles = k, collonne1=columns_bool[0],collonne2=columns_bool[1],collonne3=columns_bool[2])

    list_data3 = data3.values.tolist()

    st.sidebar.markdown("---") 

    if st.sidebar.button("Show data"):
        st.write(data3)

    st.sidebar.markdown("---")
    threshhold= st.sidebar.slider("Select a minimum threshold in percent", 0, 100, 50)
    minconfiance = st.sidebar.slider("select min confiance", 0, 100, 50,on_change=None)

    if st.sidebar.button("Apriori"):

        appriorie_dict = Apriori(list_data3,Min_Supp_percent =(threshhold/100))
        apprioried = pd.DataFrame(appriorie_dict)

        apprioried = apprioried.applymap(lambda x: 0 if not pd.to_numeric(x, errors='coerce') else x)
        apprioried['Fréquence terme'] = apprioried.sum(axis=1)
        apprioried = apprioried["Fréquence terme"]
        st.write("Fréquence terme")
        st.write(apprioried)
        rule_list = get_items(appriorie_dict)
        rules = generate_association_rules(rule_list)
        st.write("Régles possible")
        rules = pd.DataFrame(rules)
        # Define a dictionary with the new column names
        new_column_names = {0: 'Antécedent',
                            1: 'Conséquent'}
        # Rename the columns using the rename method
        rules = rules.rename(columns=new_column_names)
        st.write(rules)

        confiance, lift, cosine = calcul_confiance(rule_list, appriorie_dict,list_data3, min_confiance=minconfiance/100)

        confiance = pd.DataFrame(list(confiance.items()), columns=['Régle', 'Confiance'])
        lift = pd.DataFrame(list(lift.items()), columns=['Régle', 'Lift'])
        cosine = pd.DataFrame(list(cosine.items()), columns=['Régle', 'Cosine'])

        st.write("Mesures de correlation")
        result_df = pd.concat([confiance, lift["Lift"], cosine["Cosine"]], axis=1)
        st.write(result_df)
    st.sidebar.markdown("---")
    st.session_state.add_observation = False
    if st.sidebar.button("Ajouter une observation") or st.session_state.add_observation:
        st.session_state.add_observation = True
        st.empty()
        st.write("Ajouter une observation")
        temperature = st.number_input ("temperature",step=0.01, min_value=-20.0, max_value=60.0,value=20.0)
        humidity = st.number_input ("humidity",step=0.01, min_value=0.0, max_value=100.0,value=80.0)
        Rainfall = st.number_input ("Rainfall",step=0.01,value=220.0)
        soil = st.selectbox("Soil", data3["Soil"].unique())
        crop = st.selectbox("Crop", data3["Crop"].unique())
        Fertilizer = st.selectbox("Fertilizer", data3["Fertilizer"].unique())
        if st.button("Ajouter"):
            new_observation = {'Temperature': temperature,
                            'Humidity': humidity,
                            'Rainfall': Rainfall,
                            'Soil': soil,
                            'Crop': crop,
                            'Fertilizer': Fertilizer}
            # Append the new observation to the DataFrame
            data3.loc[len(data3)] = new_observation
            st.write("Nouvelle observation ajoutée:")
            st.write(data3.iloc[-1])  # Display the last row

            # Save the updated DataFrame to Excel
            data3.to_excel('Data/Dataset3up.xlsx', index=False)
            st.session_state.add_observation = False
           
            



def main():
    st.title("Data Preprocessing and Analysis")

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Select Dataset", ["Dataset1", "Dataset2","Dataset3"])

    if selected_page == "Dataset1":
        dataset1()
    elif selected_page == "Dataset2":
        dataset2()
    else:
        dataset3()


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()


