import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

import streamlit as st
from lib.dataset import Dataset, Dataset2
import plotly.express as px

import streamlit as st
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta 
from lib.dataset3 import Apriori, calcul_confiance, calculate_intervals, dataset_to_discret, generate_association_rules, get_items
from lib.Classification import Classification
from lib.clustering import Clustering
import pickle

data1 = pd.read_csv('Data/Dataset1.csv')
data2 = pd.read_csv('Data/Dataset2_correct.csv',index_col=0)
data3 = pd.read_excel('Data/Dataset3up.xlsx')
classificateur = pickle.load(open("./lib/classificateur.p", 'rb'))


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
    outliers_option = st.sidebar.selectbox("Handle Outliers", [None, "drop"])
    normalization_option = st.sidebar.selectbox("Normalization", [None, "minmax"])
    
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
    if st.sidebar.button("Les zones les plus impacté"):
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
    if not 'add_observation' in st.session_state:
        st.session_state.add_observation = False
    global data3
    intervale_option = st.sidebar.selectbox("methode de discrétisation", ["Brooks Carruthers", "Huntsberger","Sturges"])
    disc_option = st.sidebar.selectbox("methode de discrétisation", ["fréquence egale", "largeur egale"])
    columns = st.sidebar.multiselect("choix des colonnes", data3.columns[:3])
    data3_dicret = data3
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

        data3_dicret = dataset_to_discret(data3,intervalles_type=disc_option, nombre_intervalles = k, collonne1=columns_bool[0],collonne2=columns_bool[1],collonne3=columns_bool[2])

    st.sidebar.markdown("---") 

    if st.sidebar.button("Show data"):
        st.session_state.add_observation = False
        st.write(data3)

    st.sidebar.markdown("---")
    threshhold= st.sidebar.slider("Select a minimum threshold in percent", 0, 100, 50)
    minconfiance = st.sidebar.slider("select min confiance", 0, 100, 50,on_change=None)

    if st.sidebar.button("Apriori"):
        st.session_state.add_observation = False
            # Convertir les colonnes en type str
        data3_dicret['Rainfall']       = "R"     + data3_dicret['Rainfall'].astype(str)
        data3_dicret['Humidity']       = "H"     + data3_dicret['Humidity'].astype(str)
        data3_dicret['Temperature']    = "T"     + data3_dicret['Temperature'].astype(str)
        
        list_data3 = data3_dicret.values.tolist()
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

        confiance, lift, cosine, recommendation = calcul_confiance(rule_list, appriorie_dict,list_data3, min_confiance=minconfiance/100)

        confiance = pd.DataFrame(list(confiance.items()), columns=['Régle', 'Confiance'])
        lift = pd.DataFrame(list(lift.items()), columns=['Régle', 'Lift'])
        cosine = pd.DataFrame(list(cosine.items()), columns=['Régle', 'Cosine'])

        st.write("Mesures de correlation")
        result_df = pd.concat([confiance, lift["Lift"], cosine["Cosine"]], axis=1)
        st.write(result_df)
    st.sidebar.markdown("---")
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
            data3 = pd.read_excel('Data/Dataset3up.xlsx')
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
           
def supervised():

    st.sidebar.markdown("---")
    st.sidebar.header("Classification tests using test data")
    if st.sidebar.button("Test Decision Tree"):
        st.session_state.add_observation_superv = False
        treeResult = classificateur.testDecisionTree()

        st.write("Test Decision Tree")
        st.write("Matrice de confusion")
        mat = classificateur.confMatrix(treeResult)
        st.write(mat)
        st.write("Métriques de performance")
        metrics = classificateur.getMetrics(mat)
        st.write(pd.DataFrame(metrics))

    if st.sidebar.button("Test Knn"):
        st.session_state.add_observation_superv = False
        KnnResult = classificateur.testKnn()

        st.write("Test Knn")
        st.write("Matrice de confusion")
        mat = classificateur.confMatrix(KnnResult)
        st.write(mat)
        st.write("Métriques de performance")
        metrics = classificateur.getMetrics(mat)
        st.write(pd.DataFrame(metrics))

    if st.sidebar.button("Test Random forest"):
        st.session_state.add_observation_superv = False
        randomForestResult = classificateur.testRandomForest()

        st.write("Test Random forest")
        st.write("Matrice de confusion")
        mat = classificateur.confMatrix(randomForestResult)
        st.write(mat)
        st.write("Métriques de performance")
        metrics = classificateur.getMetrics(mat)
        st.write(pd.DataFrame(metrics))

    st.sidebar.markdown("---")
    if st.sidebar.button("Insertion de données") or st.session_state.add_observation_superv:
        classificationAlgo = st.selectbox("Select a classifier", ["Decision tree", "Knn", "Random forest"])
        st.session_state.add_observation_superv = True
        st.empty()
        st.markdown("---")
        st.write("Ajouter une observation")
        item = []
        dictitem = {}
        columns = 4
        cols = st.columns(columns)
        for nb,i in enumerate(data1.columns[:-1]):
            with cols[nb%columns]:
                item.append(st.number_input (i,step=0.01,value=0.0))
                dictitem[str(i)] = item[nb]
        st.table(dictitem)
        st.markdown("---")
        st.write("Parameters")
        oldDepth = 10
        oldminSplit = 2
        oldRandn_trees = 100
        oldRandDepth = 10
        oldRandminSplit = 2
        match classificationAlgo:
            case "Knn":
                Kinput = st.number_input("K",step=1,value=2,min_value=1)
                distanceAlgo = st.selectbox("Distance", ["euclidienne", "manhattan","minkowski","cosine","hamming"])
            case "Decision tree":
                Depth = st.number_input("Depth",step=1,value=10,min_value=1)
                minSplit = st.number_input("minSplit",step=1,value=2,min_value=2)

            case "Random forest":
                n_trees = st.number_input("n_trees",step=1,value=100,min_value=1)
                Depth = st.number_input("Depth",step=1,value=10,min_value=1)
                minSplit = st.number_input("minSplit",step=1,value=2,min_value=2)

        if st.button("predict"):
            match classificationAlgo:
                case "Decision tree":
                    if Depth != oldDepth and minSplit != oldminSplit:
                        st.write("Entrainement...")
                        classificateur.trainDecisionTree(maxDepth=Depth,minSamplesSplit=minSplit)
                        st.write("Entrainement terminé")
                    st.write("Prediction")
                    st.write(list(classificateur.tree.predict([item]))[0])
                case "Knn":
                    st.write("Prediction")
                    st.write(classificateur.knn.getClass(list(item),algo=distanceAlgo,k=Kinput))
                case "Random forest":
                    if Depth != oldRandDepth and minSplit != oldRandminSplit and n_trees != oldRandn_trees:
                        st.write("Entrainement...")
                        classificateur.trainRandomForest(n_tree=n_trees,maxDepth=Depth,minSamplesSplit=minSplit)
                        st.write("Entrainement terminé")
                    st.write("Prediction")
                    dictitem["Fertility"] = 0
                    st.write(classificateur.testRandomForest(data=pd.DataFrame(dictitem,index=[13]))[0])

def setUnsupervised(e):
    st.session_state.clust = e
def unsupervised():
    st.sidebar.markdown("---")
    null_option = st.sidebar.selectbox("Handle Null Values", ["drop", "mean"])
    outliers_option = st.sidebar.selectbox("Handle Outliers", [None, "drop", "mean", "median", "Q1Q3"])
    normalization_option = st.sidebar.selectbox("Normalization", [None, "minmax", "zscore"])
    discretize  = st.sidebar.checkbox("Discretization")
    
    ds = Dataset(data1)
    ds.preprocessData(null=null_option,outliers=outliers_option,normalisation=normalization_option)
    if discretize:
        ds.reduction(ignore=["Fertility"])
    clusterer = Clustering(ds.data)
    if st.sidebar.button("Show data"):
        st.write(clusterer.data)
    st.sidebar.markdown("---")
    if st.sidebar.button("Kmeans",on_click=setUnsupervised,args=["Kmeans"]) or st.session_state.clust =="Kmeans":
        st.write("Kmeans")
        DistanceKm = st.selectbox("Select distance kmeans", ["euclidienne", "manhattan","minkowski","cosine","hamming"])
        K = st.number_input("K",step=1,value=2,min_value=1)
        init = st.selectbox("Select init", ["random", "1/k"])
        iteration = st.number_input("iter",step=1,value=100,min_value=1)
        if st.button("run kmeans"):
            st.write("Entrainement...")
            clst,it = clusterer.kmeansClustering(K=K,itera=iteration,init=init,distance=DistanceKm)
            st.write("Entrainement terminé a l'iteration ",it)
            percent,matrice = clusterer.percent(clst)
            cols = st.columns(2)
            with cols[0]:
                st.write("Matrice de confusion")
                st.write(pd.DataFrame(matrice))
                st.write("Inter cluster distance : ",clusterer.kmeans.interClusterDistance())
                st.write("Intra cluster distance : ",clusterer.kmeans.intraClusterDistance())
            with cols[1]:
                st.write("Pourcentage de la classe dominante dans un cluster")
                st.write(pd.DataFrame(percent))
                st.write("silhouette : ",clusterer.silhouette(clst))
                st.write("calinski : ",clusterer.calinski(clst))
                st.write("davies : ",clusterer.davies(clst))

            clusterer.drawScatter(clst)
            st.pyplot()
    
    if st.sidebar.button("DBSCAN",on_click=setUnsupervised,args=["DBSCAN"]) or st.session_state.clust =="DBSCAN":
        st.write("DBSCAN")
        eps = st.number_input("eps",step=0.01,value=30.0,min_value=0.01)
        min_samples = st.number_input("min_samples",step=1,value=15,min_value=1)
        if st.button("run dbscan"):
            st.write("Entrainement...")
            clst = clusterer.dbscanClustering(eps=eps,minPts=min_samples)
            st.write("Entrainement terminé")
            percent,matrice = clusterer.percent(clst)
            cols = st.columns(2)
            with cols[0]:
                st.write("Matrice de confusion")
                st.write(pd.DataFrame(matrice))
            with cols[1]:
                st.write("Pourcentage de la classe dominante dans un cluster")
                st.write(pd.DataFrame(percent))
                st.write("silhouette : ",clusterer.silhouette(clst))
                st.write("calinski : ",clusterer.calinski(clst))
                st.write("davies : ",clusterer.davies(clst))
            
            clusterer.drawScatter(clst)
            
def main():

    st.title("Data Preprocessing and Analysis")

    st.sidebar.title("Navigation")
    selected_part = st.sidebar.radio("Select Part", ["Part1", "Part2"])
    
    if selected_part == "Part1":
        st.session_state.kmeans = False
        st.session_state.dbscan = False

        selected_page = st.sidebar.radio("Select Dataset", ["Dataset1", "Dataset2","Dataset3"])
        if selected_page == "Dataset1":
            st.session_state.add_observation_superv = False
            dataset1()
        elif selected_page == "Dataset2":
            st.session_state.add_observation_superv = False
            dataset2()
        else:
            dataset3()
    else:

        selected_type = st.sidebar.radio("Select type", ["Analyse supervisée", "Analyse non supervisée"])
        if selected_type == "Analyse supervisée":
            st.session_state.kmeans = False
            st.session_state.dbscan = False
            st.write("Analyse supervisée")
            supervised()
        else:
            if "clust" not in st.session_state:
                st.session_state.clust = ""
            st.session_state.add_observation_superv = False
            st.write("Analyse non supervisée")
            unsupervised()

if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()