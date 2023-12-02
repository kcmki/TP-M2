
import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import itertools
import math
import pandas as pd

import math

def calculate_intervals(n, method='Brooks Carruthers'): # n = nombre d'observations

    if method == 'Brooks Carruthers':
        K = int(5 * math.log10(n))
    elif method == 'Huntsberger':
        K = int(1 + (math.log10(n) * 10 / 3))
    elif method == 'Sturges':
        K = int(math.log2(n + 1))

    return K

def interval_amplitude(data):
    return data.max() - data.min()


def discretisation_par_amplitude_column(data, nombre_intervalles ):
    intervalles = []
    intervalles.append(data.min())
    for i in range(1, nombre_intervalles):
        a = data.min() + i * interval_amplitude(data) / nombre_intervalles
        a = round(a, 2)
        intervalles.append(a)
    intervalles.append(data.max())
    return intervalles


def discretisation_par_taille_column(data, nombre_intervalles):
    intervalles = []
    intervalles.append(data.min())
    for i in range(1, nombre_intervalles):
        a = data.quantile(i / nombre_intervalles)
        a = round(a, 2)
        intervalles.append(a)
    intervalles.append(data.max())
    return intervalles


def dataset_to_discret(data, intervalles_type='fréquence egale', nombre_intervalles=10, collonne1=False, collonne2=False, collonne3=False):
    collones_to_discretise = []
    if collonne1:
        collones_to_discretise.append('Rainfall')
    if collonne2:
        collones_to_discretise.append('Humidity')
    if collonne3:
        collones_to_discretise.append('Temperature')

    for collone in collones_to_discretise:
        if intervalles_type == 'largeur egale':
            intervalles = discretisation_par_amplitude_column(data[collone], nombre_intervalles)
        else:
            intervalles = discretisation_par_taille_column(data[collone], nombre_intervalles)
        
        print("colonne ", collone)
        print("intervalles ", intervalles)

        for i in range(len(data)):
            for j in range(len(intervalles)):
                if float(data.loc[i, collone]) <= intervalles[j]:
                    data.loc[i, collone] = intervalles[j]
                    break  # Ajout de cette instruction pour éviter la comparaison inutile après avoir trouvé l'intervalle approprié

    # Convertir les colonnes en type str
    data['Rainfall'] = data['Rainfall'].astype(str)
    data['Humidity'] = data['Humidity'].astype(str)
    data['Temperature'] = data['Temperature'].astype(str)

    print(data)
    return data


def transaction_to_item(transaction_data):
    # Crée un dictionnaire pour compter le nombre d'occurrences de chaque élément
    dict_item = {}

    for transaction in transaction_data:
        for item in transaction:
            # Vérifier si l'élément est une chaîne de caractères
            if isinstance(item, float):
                # Utiliser lower() seulement pour les chaînes de caractères
                item = str(item).lower()

            # Convertir l'élément en chaîne de caractères si ce n'est pas déjà le cas
            item = str(item)

            if item in dict_item:
                dict_item[item] += 1
            else:
                dict_item[item] = 1
    
    dict_item = {cle: dict_item[cle] for cle in sorted(dict_item , key=str.lower)}
    return dict_item


def filtre_dictionnaire(dictionnaire, Min_Supp):
    # Crée un nouveau dictionnaire pour stocker les éléments filtrés
    dictionnaire_filtré = {}

    for clé, valeur in dictionnaire.items():
        if valeur >= Min_Supp:
            dictionnaire_filtré[clé] = valeur

    return dictionnaire_filtré

def tuple_to_item(tuples):

    tuplee = list(tuples.keys())[0]

    if (type(tuplee) == tuple):
        tuples = list(tuples.keys())
        items = []
        for one_tuple in tuples:
            for item in one_tuple:

                if item not in items:
                    items.append(item)  
    else:
        items = []
        for item in tuples:
            if item not in items:
                items.append(item)  

    return items

def items_to_tuple(Items, Taille_tuple):
    Items.sort(key=str.lower)
    all_tuples = list(itertools.combinations(Items, Taille_tuple))

    dictionnaire = {}
    for one_tuple in all_tuples:
        dictionnaire[one_tuple] = 0

    return dictionnaire

def check_if_line_contains_elements(data, elements):
    for i in elements:
        try : 
            i = float (i)
        except : 
            i = i 
        count = data.count(i)
        if count == 0:
            return False
    return True

    
def update_frequence(transaction_data, tuples):
    Lk = tuples
    for ligne in transaction_data:
        for (item, valeur) in Lk.items():
            if check_if_line_contains_elements(ligne,item):
                Lk[item] += 1
    return Lk

def Apriori(transaction_data, Min_Supp_number = -1, Min_Supp_percent = -1):
    
    if (Min_Supp_number == -1 and Min_Supp_percent == -1):
        print("Erreur: Veuillez entrer un seuil")
        return -1
    
    if (Min_Supp_number != -1):
        Min_Supp = Min_Supp_number

    if (Min_Supp_percent != -1):
        Min_Supp = len(transaction_data)*Min_Supp_percent

    phase = 1
    Lk = {}
    Ck = {}
    Ck[1] = {}
    Lk[1] = {}

    # Phase 1
    Ck[phase] =     transaction_to_item(transaction_data)
    #print ("Ck["    ,phase, "]\t", Ck[phase])
    Lk[phase] =     filtre_dictionnaire(Ck[phase], Min_Supp)
    #print ("Lk["    ,phase, "]\t", Lk[phase])

    #debut boucle while

    while (Lk[phase] != {}):

        phase += 1

        #Générer toutes les combinaisons possibles formant des k-itemsets

        items       =   tuple_to_item (Lk[phase-1])
        Ck[phase]   =   items_to_tuple(items, phase)
        #print ("Ck["    ,phase, "]\t", Ck[phase])
        Ck[phase] =     update_frequence    (transaction_data, Ck[phase])

        if (len(Ck[phase]) == 0):
            break

        #print ("Ck["    ,phase, "]\t", Ck[phase])
        temporary =     filtre_dictionnaire (Ck[phase], Min_Supp)
        
        if (len(temporary) == 0):
            #print ("Fin de apriori")
            break
        
        Lk[phase] = temporary
        #print ("Lk["    ,phase, "]\t", Lk[phase])
    return Lk

def calcul_taille_tuple(one_tuple):
    
    if not isinstance(one_tuple, (list, tuple  )):
        return 1
    return len(one_tuple)


def tuple_to_items(tuples):
    items = []
    if calcul_taille_tuple(tuples) == 1:
        items.append(tuples)
    
    else:
        for i in tuples:
            items.append(i)  

    return items


def ont_elements_en_commun(liste1, liste2):
    for element in liste1:
        if element in liste2:
            return True
    return False

def get_items(result_dict):

    k_items = []

    for cle in range (1, len(result_dict)+1):
        for item in result_dict[cle]:
            if item not in k_items:
                k_items.append(item)
    return k_items

def generate_association_rules(k_items):
    rules = []

    for antecedent in k_items:
        for consequent in k_items:
            # Vérifier si l'antécédent et la conséquence sont différents
            if antecedent != consequent and not ont_elements_en_commun(tuple_to_items(antecedent), tuple_to_items(consequent) ):
                # Vérifier si l'antécédent et la conséquence ne partagent aucun item ou partie d'item
                # Ajouter la règle si toutes les conditions sont remplies
                rules.append((antecedent, consequent))

    # Afficher les règles générées
    """for rule in rules:
        print(rule[0], '->', rule[1])"""

    return rules

def union_items_tuples(tuple1, tuple2):
    items = []
    
    if calcul_taille_tuple(tuple1) == 1:
        items.append(tuple1)
    else:
        for i in tuple1:
            items.append(i)
    
    if calcul_taille_tuple(tuple2) == 1:
        items.append(tuple2)
    else:
        for i in tuple2:
            items.append(i)
    
    items.sort(key=str.lower)
    return items


def calculate_metrics(support_AB, support_A, support_B, total_transactions, size_A, size_B):

    confidence = support_AB / support_A if support_A != 0 else 0

    lift_denominator = (support_A / total_transactions) * (support_B / total_transactions)
    lift = support_AB / lift_denominator if lift_denominator != 0 else 0

    cosine_numerator = support_AB
    cosine_denominator = math.sqrt(support_A * support_B) * math.sqrt(size_A * size_B)
    cosine_similarity = cosine_numerator / cosine_denominator if cosine_denominator != 0 else 0

    return confidence, lift, cosine_similarity

def calcul_confiance(rules, result_dict,list_data3, min_confiance = 0.1):

    confiance = {}
    lift = {}
    cosine = {}

    for rule in rules:
        
        A = rule[0]
        B = rule[1]
        A_u_B = union_items_tuples(A, B)
        
        if calcul_taille_tuple(A_u_B) <= len(result_dict):

            
            temp = result_dict[calcul_taille_tuple(A_u_B)]
            # Convert A_u_B to a tuple before using it as a key
            A_u_B_tuple = tuple(A_u_B)


            if A_u_B_tuple in temp.keys():
                Supp_A = result_dict[calcul_taille_tuple(A)][A]
                Supp_A_u_B = result_dict[calcul_taille_tuple(A_u_B_tuple)][A_u_B_tuple]

                calcul =  calculate_metrics(Supp_A_u_B, Supp_A, Supp_A, len(list_data3), calcul_taille_tuple(A), calcul_taille_tuple(B))
                if (calcul [0] >= min_confiance):
                    confiance[A, B],lift[A, B], cosine[A, B]  = calcul


                


    return confiance, lift, cosine
        

def afficher_lift (lift, seuille):
    print ("Regle lift " )
    for i in lift:
        if lift[i] > seuille:
            print (i[0],'->',i[1],':', lift[i])


def afficer_cosine(cosine,seuille = 0.5):
    print ("Regle cosine " )
    for i in cosine:
        if cosine[i] > seuille:
            print (i[0],'->',i[1],':', cosine[i])


def afficher_regles_confiance(confiance, Min_Conf):
    for i in confiance:
        if confiance[i] >= Min_Conf:
            print (i[0],'->',i[1],':', confiance[i])

def nombre_regle_confiane(confiance, Min_Conf):
    nombre = 0
    for i in confiance:
        if confiance[i] >= Min_Conf:
            nombre += 1
    return nombre
