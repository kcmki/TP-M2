import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.dates as mdates



class Dataset:
    def __init__(self, data):

        self.default = data.copy()
        self.data = data
        self.tendance = None
        self.symetrie = None
        self.boxes = None
        self.correlation = None
    def toNumeric(self):
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

    def deleteNull(self,dtype="drop"):
        if dtype == "drop":
            self.data = self.data.apply(pd.to_numeric, errors='coerce')
            self.data = self.data.dropna()
        elif dtype == "mean":
            self.data = self.data.apply(pd.to_numeric, errors='coerce')
            self.data = self.data.fillna(self.data.mean())
    
    def mean(self):
        return self.data.mean(numeric_only=True)

    def median(self):
        return self.data.median(numeric_only=True)

    def mode(self):
        mode = self.data.mode(numeric_only=True)
        return mode.mean()
    
    def tendances(self):
        print(self.data)
        dictio = {"mean":self.data.mean(),"median":self.data.median(),"mode":self.data.mode()}
        if dictio["mode"].shape != dictio["mean"].shape:
            dictio["mode"] = dictio["mode"].mean()
        print(dictio)
        self.tendance = pd.DataFrame(dictio)
        print(self.tendance)

    def defSymetrie(self):
        self.symetrie = {}

        self.tendances() 
        
        for i in self.tendance.index:
            if self.tendance['mean'][i] -  self.tendance['median'][i] > (self.tendance['median'][i] * 0.05):
                self.symetrie[i] = "positive"
            elif self.tendance['mean'][i] - self.tendance['median'][i] < -(self.tendance['mean'][i] * 0.05):
                self.symetrie[i] = "negative"
            else:
                self.symetrie[i] = "symetric"
        self.symetrie = pd.DataFrame(list(self.symetrie.items()), columns=['Element', 'Property'])
        
        return self.symetrie
    
    def setBoxes(self):
        self.boxes = []
        print(self.data.columns)
        for column in self.data.columns:

            box = self.data.boxplot(column=column,return_type='dict')
            self.boxes.append((column,box))
        plt.clf()
    
    def drawBoxes(self,col=None):

        if col in self.data.columns:
            plt.figure()
            plt.title("boxplot of "+col)   
            dictD = self.data.boxplot(column=col,return_type='dict')
            print("données aberrantes de "+col)
            print(dictD['fliers'][0].get_ydata()) 
            plt.show()
            return None
        for col in self.data.columns:
            plt.figure()
            plt.title("boxplot of "+col)   
            dictD = self.data.boxplot(column=col,return_type='dict')
            print("données aberrantes de "+col)
            print(dictD['fliers'][0].get_ydata()) 
            plt.show()

    def drawHist(self,col=None):
        if col in self.data.columns:
            plt.figure(figsize=(3, 2))
            self.data.hist(column=col)
            plt.title(f'Histogram of {col}')
            return None
        for column in self.data.columns:  
            plt.figure(figsize=(3, 2))
            self.data.hist(column=column,)
            plt.title(f'Histogram of {column}')

    def deleteOutliers(self,dtype="drop"):
        if not self.boxes:
            self.setBoxes()
        if dtype == "drop":
            
            for i in range(len(self.boxes)):
                for outlier in self.boxes[i][1]['fliers'][0].get_ydata():
                    self.data = self.data[ self.data[self.boxes[i][0]] != outlier] 
        elif dtype == "mean":
            if not self.tendance:
                self.tendances()
            for i in range(len(self.boxes)):
                for outlier in self.boxes[i][1]['fliers'][0].get_ydata():
                    self.data.loc[self.data[self.boxes[i][0]] == outlier, self.boxes[i][0]] = self.tendance["mean"][i]
        elif dtype == "median":
            if not self.tendance:
                self.tendances()
            for i in range(len(self.boxes)):
                for outlier in self.boxes[i][1]['fliers'][0].get_ydata():
                    self.data.loc[self.data[self.boxes[i][0]] == outlier, self.boxes[i][0]] = self.tendance["median"][i]
        elif dtype == "Q1Q3":
            for i in range(len(self.boxes)):
                for outlier in self.boxes[i][1]['fliers'][0].get_ydata():
                    if outlier > self.data[self.boxes[i][0]].mean():
                        self.data.loc[self.data[self.boxes[i][0]] == outlier, self.boxes[i][0]] = self.boxes[i][1]['whiskers'][1].get_ydata()[0]
                    if outlier < self.data[self.boxes[i][0]].mean():
                        self.data.loc[self.data[self.boxes[i][0]] == outlier, self.boxes[i][0]] = self.boxes[i][1]['whiskers'][0].get_ydata()[0]
    
    def normalisation(self,dtype="minmax"):
        if dtype == "minmax":
            self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        elif dtype == "zscore":
            self.data = (self.data - self.data.mean()) / self.data.std()

    def setCorrelation(self,i=None,j=None):
        if i in self.data.columns and j in self.data.columns:
            return self.data[i].corr(self.data[j])
        
        self.correlation = self.data.corr()
        return self.correlation
    
    def dropCorrelation(self,threshold=0.8):

        if type(self.correlation) == type(None):
            self.setCorrelation()
        # Identify pairs of columns with correlation above the threshold
        high_corr_pairs = []
        for i in range(len(self.correlation.columns)):
            for j in range(i+1, len(self.correlation.columns)):
                if abs(self.correlation.iloc[i, j]) > threshold:
                    high_corr_pairs.append((self.correlation.columns[i], self.correlation.columns[j]))

        print(len(high_corr_pairs))
        # Drop one column from each highly correlated pair
        for col1, col2 in high_corr_pairs:
            # Drop the column with the higher correlation with other columns
            if col1 in self.data.columns and col2 in self.data.columns and col1 != "Fertility" and col2 != "Fertility":
                if abs(self.correlation[col1].mean()) > abs(self.correlation[col2].mean()):
                    self.data = self.data.drop(col1, axis=1)
                else:
                    self.data = self.data.drop(col2, axis=1)

    def Scatter(self,i=None,j=None):

        if i in self.data.columns and j in self.data.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(self.data[i], self.data[j])
            plt.xlabel(i)
            plt.ylabel(j)
            plt.title(f'Scatter plot of {i} and {j}')
            plt.show()
            return None
        
        for i in self.data.columns:
            for j in self.data.columns[self.data.columns.get_loc(i)+1:]:
                plt.figure(figsize=(6, 4))
                plt.scatter(self.data[i], self.data[j])
                plt.xlabel(i)
                plt.ylabel(j)
                plt.title(f'Scatter plot of {i} and {j}')
                plt.show()

    def preprocessData(self,null="drop",outliers="drop",normalisation="minmax"):
        self.deleteNull(dtype=null)
        
        if outliers:
            self.deleteOutliers(dtype=outliers)
        
        if normalisation:
            self.normalisation(dtype=normalisation)
    
    def Discretisation(self,nbIntervalle,ignore=None):
        if not ignore:
            ignore = []

        for i in self.data.columns:
            if i not in ignore:
                bins = []
                for j in range(nbIntervalle):
                    taille_inter = (self.data[i].max() - self.data[i].min()) / nbIntervalle
                    bins.append(self.data[i].min() + (taille_inter * j))
                self.data[i] = pd.cut(self.data[i],nbIntervalle,labels=bins)
    
    def reduction(self,ignore=None):
        self.dropCorrelation()

        k = int(1 + 3.3 * np.log10(len(self.data)))
        self.Discretisation(k,ignore=ignore)
        self.data = self.data.drop_duplicates(subset=self.data.columns[:-1])
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
    def resetData(self):
        self.data = self.default.copy()


class Dataset2(Dataset):

    def deleteNull(self,dtype="drop",ignore=None):
        if dtype == "drop":
            for column in self.data.columns:
                if column not in ignore:
                    self.data[column] = self.data[column].apply(pd.to_numeric, errors='coerce')
                    self.data[column] = self.data[column].dropna()
        elif dtype == "mean":
            for column in self.data.columns:
                if column not in ignore:
                    self.data[column] = self.data[column].apply(pd.to_numeric, errors='coerce')
                    self.data[column] = self.data[column].fillna(self.data[column].mean())

        self.data = self.data.dropna()   

    def toDate(self,col="Date"):
        self.data[col] = pd.to_datetime(self.data[col], format='mixed')
    
    def deleteOutliers(self,dtype="drop",ignore=None):
            if not self.boxes:
                self.setBoxes(ignore=ignore)
            if dtype == "drop":
                
                for i in range(len(self.boxes)):
                    for outlier in self.boxes[i][1]['fliers'][0].get_ydata():
                        if outlier not in ignore:
                            self.data = self.data[ self.data[self.boxes[i][0]] != outlier] 
            elif dtype == "mean":
                if not self.tendance:
                    self.tendances()
                for i in range(len(self.boxes)):
                    for outlier in self.boxes[i][1]['fliers'][0].get_ydata():
                        if outlier not in ignore:
                            self.data.loc[self.data[self.boxes[i][0]] == outlier, self.boxes[i][0]] = self.tendance["mean"][i]
            elif dtype == "median":
                if not self.tendance:
                    self.tendances()
                for i in range(len(self.boxes)):
                    for outlier in self.boxes[i][1]['fliers'][0].get_ydata():
                        if outlier not in ignore:
                            self.data.loc[self.data[self.boxes[i][0]] == outlier, self.boxes[i][0]] = self.tendance["median"][i]
            elif dtype == "Q1Q3":
                for i in range(len(self.boxes)):
                    for outlier in self.boxes[i][1]['fliers'][0].get_ydata():
                        if outlier not in ignore:
                            if outlier > self.data[self.boxes[i][0]].mean():
                                self.data.loc[self.data[self.boxes[i][0]] == outlier, self.boxes[i][0]] = self.boxes[i][1]['whiskers'][1].get_ydata()[0]
                            if outlier < self.data[self.boxes[i][0]].mean():
                                self.data.loc[self.data[self.boxes[i][0]] == outlier, self.boxes[i][0]] = self.boxes[i][1]['whiskers'][0].get_ydata()[0]
    
    def normalisation(self,dtype="minmax",ignore=None):
            if dtype == "minmax":
                for column in self.data.columns:
                    if column not in ignore:
                        self.data[column] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())
            elif dtype == "zscore":
                for column in self.data.columns:
                    if column not in ignore:
                        self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

    def setBoxes(self,ignore=None):
        self.boxes = []
        for column in self.data.columns:
            if column not in ignore:
                box = self.data.boxplot(column=column,return_type='dict')
                self.boxes.append((column,box))
        plt.clf()

    def drawScatter(self,i=None,j=None):
        if i in self.data.columns and j in self.data.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(self.data[i], self.data[j])
            plt.xlabel(i)
            plt.ylabel(j)
            plt.title(f'Scatter plot of {i} and {j}')
            plt.show()
            return None
        
        for i in self.data.columns:
            for j in self.data.columns[self.data.columns.get_loc(i)+1:]:
                plt.figure(figsize=(6, 4))
                plt.scatter(self.data[i], self.data[j])
                plt.xlabel(i)
                plt.ylabel(j)
                plt.title(f'Scatter plot of {i} and {j}')
                plt.show()

    def barGraph(self,i=None,j=None,title=None):
        if i in self.data.columns and j in self.data.columns:
            plt.figure(figsize=(6, 4))
            plt.bar(self.data[i], self.data[j])
            plt.xlabel(i)
            plt.ylabel(j)
            if title:
                plt.title(title)
            else:    
                plt.title(f'Bar plot of {i} and {j}')
            plt.show()
            return None
        
    def correctDates(self):

        for i in range(len(self.data)):
            for j in range(i,len(self.data)):
                if self.data["time_period"][i] == self.data["time_period"][j]:

                    self.data["Start date"][j] = self.data["Start date"][i]
                    self.data["end date"][j] = self.data["end date"][i]
                    
        for i in range(len(self.data)): 
            try:    
                self.data['end date'][i] = pd.to_datetime(self.data['end date'][i] )
                self.data['Start date'][i]  = pd.to_datetime(self.data['Start date'][i] )
            except Exception as e:
                rows = self.data[self.data['time_period']==self.data['time_period'][i]+1]
                year = rows["end date"].iloc[0].year
                
                date = self.data['Start date'][i].split("-")
                self.data['Start date'][i] = pd.to_datetime(str(year)+'-'+date[1]+'-'+date[0],format="%Y-%b-%d")
                date = self.data['end date'][i].split("-")
                self.data['end date'][i] = pd.to_datetime(str(year)+'-'+date[1]+'-'+date[0],format="%Y-%b-%d")


        self.data['Start date'] = pd.to_datetime(self.data['Start date'])
        self.data['end date'] = pd.to_datetime(self.data['end date'])

    def correctDates2(self):
        modified_data = self.data.copy()

        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                if self.data["time_period"][i] == self.data["time_period"][j]:
                    modified_data.at[j, "Start date"] = self.data["Start date"][i]
                    modified_data.at[j, "end date"] = self.data["end date"][i]

        # Update the original DataFrame with modified data
        self.data = modified_data
        print(modified_data["Start date"])


    def preprocessData(self,null="drop",outliers="drop",normalisation="minmax",ignore=None):
        
        self.deleteNull(dtype=null,ignore=ignore)
        
        if outliers:
            self.deleteOutliers(dtype=outliers,ignore=ignore)
        
        if normalisation:
            self.normalisation(dtype=normalisation,ignore=ignore)

    def YearlyDistributionByZoneTests(self,zone=0):
        data = {"zone":self.data['zcta'],"date":self.data['Start date'], "test":self.data['test count'],'case count':self.data["case count"],"positive tests":self.data['positive tests']}

        df = pd.DataFrame(data)
        df = df[df['zone'] == zone]
        df.sort_values(by=['date'], inplace=True)


        plt.plot(df["date"],df["test"], label='Tests Conducted', marker='')
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.title(f'Yearly Distribution of COVID-19 tests in zone {zone}')
        plt.xticks(fontsize=8, rotation = 90)
        plt.show()

    def YearlyDistributionByZonePTC(self,zone=0):
        data = {"zone":self.data['zcta'],"date":self.data['Start date'], "test":self.data['test count'],'case count':self.data["case count"],"positive tests":self.data['positive tests']}

        df = pd.DataFrame(data)
        df = df[df['zone'] == zone]
        df.sort_values(by=['date'], inplace=True)

        plt.plot(df["date"],df["case count"], label='case count', marker='')
        plt.plot(df["date"],df["positive tests"], label='positive tests count', marker='')

        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.title(f'Yearly Distribution of COVID-19 positive tests count and cases in zone {zone}')

        plt.xticks(fontsize=8, rotation = 90)
        plt.show()
        
    def barYearCase(self):
        
        self.data.set_index(self.data['Start date'].dt.year, inplace=True)

        plt.figure(figsize=(8, 5))
        self.data.groupby(self.data.index)['case count'].sum().plot(kind='bar', color='skyblue',stacked=True)
        plt.title('Yearly Distribution of Positive COVID-19 Cases')
        plt.xlabel('Year')
        plt.ylabel('Positive Cases')
        plt.show()
    def barZoneCase(self):
        self.data.set_index(self.data['zcta'], inplace=True)

        plt.figure(figsize=(8, 5))
        self.data.groupby(self.data.index)['case count'].sum().plot(kind='bar', color='skyblue',stacked=True)
        plt.title('Zone Distribution of Positive COVID-19 Cases')
        plt.xlabel('zone')
        plt.ylabel('Cases count')
        plt.show() 
    def mostAffectedZone(self,count=5):
        self.data.set_index(self.data['zcta'], inplace=True)

        plt.figure(figsize=(8, 5))
        self.data.groupby(self.data.index)['case count'].sum().sort_values(ascending=False).head(count).plot(kind='bar', color='skyblue')
        plt.title('Yearly Distribution of Positive COVID-19 Cases')
        plt.xlabel('Year')
        plt.ylabel('Positive Cases')
        plt.show()
    
    def testByPopu(self):
        self.data.set_index(self.data['population'], inplace=True)

        plt.figure(figsize=(8, 5))
        self.data.groupby(self.data.index)['test count'].sum().plot(kind='line', color='skyblue',stacked=True)
        plt.title('test count by population size')
        plt.xlabel('population')
        plt.ylabel('test count')
        plt.show()
    
    def CaseCountPosTestByZone(self,Zone=0,start=pd.to_datetime('2019-01-01'),end=pd.to_datetime('2024-01-01')):
        df = self.data[(self.data["Start date"] > start) & (self.data["Start date"] < end) & (self.data["zcta"] == Zone)]
        
        plt.scatter(df["case count"],df["positive tests"])
        plt.xlabel("case count")
        plt.ylabel("positive tests")
        plt.title("case count vs positive tests in zone "+str(Zone))
        plt.show()

    def posTestByTestCountByZone(self,Zone=0,start=pd.to_datetime('2019-01-01'),end=pd.to_datetime('2024-01-01')):
        df = self.data[(self.data["Start date"] > start) & (self.data["Start date"] < end) & (self.data["zcta"] == Zone)]

        plt.scatter(df["positive tests"],df["test count"])
        plt.xlabel("positive tests")
        plt.ylabel("test count")
        plt.title("positive tests vs test count in zone "+str(Zone))
        plt.show()

    def uniquesData(self,col):
        # this function changes big values unique data to small values it does some sort of discretisation 
        # like 9400 and 2200 who are post codes which the values doesn't really have a meaning in calculations turns into 0 and 1
        if col in self.data.columns:
            uniqueVals = self.data[col].unique()
            for i in range(len(uniqueVals)):
                self.data.loc[self.data[col] == uniqueVals[i], col] = i