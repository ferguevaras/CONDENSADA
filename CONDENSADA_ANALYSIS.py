import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
class condensada_analysis():
  
  def __init__(self):
    #Setting default vars for our analysis
    #Choose your file path
    self.path = '/content/drive/MyDrive/Condensada/Raw_data/LECHE_CONDENSADA.csv'
    self.mercado = 'TOTAL AUTOS SCANNING MEXICO'
    self.year = 2021
    self.df=''
    self.tracklist=['PRONTO','LA LECHERA','Propia']
  #method for read file and validate columns
  def read_file_validation(self):
    df = pd.read_csv(self.path)
    print(f'-----Dim of Data Frame:{df.shape}-----')
    clmns_val = ['MERCADO', 'PROD_TAG', 'DESCRIPCION', 'CATEGORIA', 'FABRICANTE',
       'MARCA', 'ENVASE', 'SUBTIPO', 'PesoVolUnitario', 'ITEM', 'SEM', 'MES',
       'ANIO', 'VentasUnidades', 'VentasValor', 'VentasUnidadesEQ',
       'DistribucionNumerica', 'DistribucionPonderada', 'PrecioPromedio',
       'MESNUM']
    #Check if the list of columns are the same in the new file 
    check =  all(item in df.columns.tolist() for item in clmns_val)
    if check is False:
        print('Caution, the columns arent the same of the past analysis')  
    else :
        print("*****Columns all right!!*****")
        self.df=df
       
  def wrangling(self):
    #We filter columns with the default values
    self.df=self.df[(self.df['MERCADO']==self.mercado) & (self.df['ANIO']==self.year)]
    self.transform()
    self.df=self.df[self.df['MARCA'].isin(self.tracklist)]
    self.assign_date()
    #Function that creates all the dates in the dataframe cuz at the begining there are only month and week number
  def assign_date(self):
    def create_date(fila):
      import datetime
      d = f'{fila.ANIO}-W{fila.Sem_numb}'
      r = datetime.datetime.strptime(d + '-1', '%G-W%V-%u')
      return r

    self.df['date']=self.df.apply(create_date,axis=1)
    print('*****Dates generated succesfully*****')

  def transform(self):
    #We add a new column called sem_numb, extracting the number of month in SEM
    self.df['Sem_numb']=self.df["SEM"].dropna().str.extract(r"(\d{2})").astype(int)
    #Setting 'Propia' Value in FABRICANTE col, where the row value is equal to Controlled Label
    self.df.loc[self.df['FABRICANTE'] == 'CONTROLLED LABEL', ['MARCA']] = 'Propia'
    self.df['precio']=self.df['VentasValor']/self.df['VentasUnidades']
  
  def load(self):
    self.df = self.df[['date','DESCRIPCION','MARCA','ITEM','VentasValor','VentasUnidades','precio','SEM']]
    df_final = self.df.groupby(['date','DESCRIPCION','ITEM','MARCA']).agg({'VentasValor':'mean','VentasUnidades':'mean','precio':'mean','SEM':'count'}).reset_index()
    #df_def=self.test_data_points(df_final)
    return df_final
    #check if the data has enough points for the analysis and plot demmand curves
  def test_data_points(self,df_final):
    items=df_final['ITEM'].unique().tolist()
    lidf=[]
    for r in items:
      dfs=df_final[df_final['ITEM']==r]
      if len(dfs)>=27:
        print(dfs.shape)
        print(dfs['DESCRIPCION'].iloc[0],'|',r)
        import seaborn as sns; sns.set_theme(color_codes=True)
        ax = sns.regplot(x="precio", y="VentasValor", data=dfs).set_title(dfs['DESCRIPCION'].iloc[0])
        plt.figure() 
        sns.lineplot(x='date', y= 'VentasValor',
                data=dfs
                ).set_title(dfs['DESCRIPCION'].iloc[0]) 
        plt.xticks(rotation=45)
        plt.figure()
        lidf.append(dfs) 
    df_sequeda=pd.concat(lidf)
    return df_sequeda
  #take the control of the deployment %MLOPS
  def deploy(self):
    self.read_file_validation()
    self.wrangling()
    return self.load()
########-------------------------------------------------------------------------------------------------##############################################
#Create class correlaciones that help us to transform data 
class correlaciones():
  def __init__(self,df):
    self.df = df
    #Skus to filter
    self.skus = ['NESTLE LA LECHERA CONDENSADA DOYPACK 209 GR 7501058631961','NESTLE LA LECHERA ORIGINAL DOY PACK 670GR=507,58ML, 7506475104470','LA LECHERA ORIGINAL LECHE CONDENSADA LATA 375GR=284,1ML, 7506475104722']
    self.df_lacat=''
  #Summarizing the sell out of LA LECHERA
  def summarize_total(self):
    self.df_cat=self.df[self.df['MARCA']=='LA LECHERA'][['date','DESCRIPCION','ITEM','MARCA','VentasValor','VentasUnidades','precio','SEM']]
    self.df_lacat=self.df_cat.groupby(['date'])['VentasValor'].sum().reset_index()
    self.df_lacat.columns=['date','LA LECHERA']
  #Filtering data and take only the expected SKU's
  def filter_skus(self):
    self.df=self.df[self.df['DESCRIPCION'].isin(self.skus)]
  #Transforming to a Panel Data
  def panel_data(self):
    result=self.df.pivot_table(index=['date'], 
                        columns='DESCRIPCION', 
                        values=['VentasValor'], aggfunc=np.sum)
    result.columns = result.columns.droplevel()
    resultnew=pd.DataFrame(result.to_records()).iloc[:, :]
    df_det=resultnew.merge(self.df_lacat,on='date',how='right')
    df_silver=df_det.tail(8)
    df_silver=df_silver.iloc[:, 1:] # Primera columna
    self.create_spearman(df_silver)
    return df_silver
  #Spearman Method
  def create_spearman(self,df2paracor):
    df2corr=df2paracor.corr(method='spearman')
    df2corr.to_csv('Corr_Condensada_LALECHERA.csv',encoding='utf-8')
    plt.subplots(figsize=(10,6))
    sns.heatmap(df2corr,cmap='coolwarm',annot=True)
  #take the control of the deployment %MLOPS
  def deploy_model(self):
    self.summarize_total()
    self.filter_skus()
    return self.panel_data()
########-------------------------------------------------------------------------------------------------##############################################33
#create class elasticities that will help us preparing the data for the reg
class elasticities():
  def __init__(self,df):
    self.df = df
    #selecting only 'lata' SKUS
    self.list_lata = ['CONTROLLED LABEL','LA LECHERA CHIQUITA NESTLE SEMIDESCREMADA LATA  100.0GR 7501059211209','LA LECHERA NESTLE LECHE CONDENSADA DESLACTOSADA LATA 384GR 7501058619976','NESTLE LA LECHERA DULCE DE LECHE LATA 370GR/290.65ML 7501058619563','NESTLE LA LECHERA LECHE CONDENSADA LATA 387GR NAL 7501058617873','NESTLE LA LECHERA LIGHT LATA 397GR/293.42ML 7501058619570','PRONTO LECHE CONDENSADA LATA 380GRS 1PZA 7501200486005']
  #Checking lata Filter and suit up the data before reg
  def check_lata(self):
    self.df=self.df[self.df['DESCRIPCION'].isin(self.list_lata)]
    self.df.columns=['date','name','upc','marca','valor','kilo','precio','count']
    self.df["date"] = pd.to_datetime(self.df["date"], format='%Y-%m')
    self.df['name']=self.df['name'].astype(str)
    return self.df 
  #Take the control of the deployment %MLOPS
  def deploy_elasticities(self):
    return self.check_lata()

####### SETTING THE REG METHOD OUT THE POO###########################################3333

def make_regression(df):
  df=df
  df.dropna(inplace=True)
  ch = df.assign(pricesales = df['precio']*df.kilo,
                    valorsales = df['valor']*df.kilo)

  ch_agg = ch.groupby(['date','name'])[['kilo','pricesales','valorsales']].sum()
  ch_agg['precio'] = ch_agg.pricesales/ch_agg.kilo
  ch_agg['valor'] = ch_agg.valorsales/ch_agg.kilo
  ch_agg = ch_agg.reset_index().sort_values('date')
  ch_agg['kilo_lw'] = ch_agg.groupby(['name'])['kilo'].shift(1)

  def clean(df,p):
    df_p = df.query('name == @p').drop(['name','precio','pricesales','valor','valorsales'], axis = 1)
    prices = pd.pivot_table(df[['date','name','precio']], values='precio', index=['date'],columns=['name'], aggfunc=np.sum).reset_index()
    prices.columns = ['date']+ ['P_' + x for  x in prices.columns[1:]]
    prices['t'] = np.exp(np.array([*range(0,prices.shape[0])]))
    return pd.merge(df_p, prices, how = 'left',on = 'date')

  marcas = ch_agg.name.unique()
  list_train_df = {p:clean(ch_agg,p) for p in marcas}

  import statsmodels.api as sm
  def ols_fit_params(df,name):
    try:
      print(name)
      dum=list_train_df[name].drop(['date'],axis=1).apply(np.log, axis = 0).dropna(axis=0).copy()
      y=dum.kilo.copy()
      X=dum.drop(['kilo'],axis = 1).copy()
      X2 = sm.add_constant(X)
      est = (sm.OLS(y, X2)
                  .fit()
                  .params
                  .reset_index()
                  .rename(columns={'index':'type',0:'parameter'})
                  )
      results=sm.OLS(y, X2).fit()
      print(results.summary())
      
      return est
      print(est)
    except: 
      print('oh,oh')
      return None

  parameters_dic = {x: ols_fit_params(list_train_df[x], x)  for x in list_train_df.keys()}
  parametros=pd.concat(parameters_dic).reset_index().drop('level_1', axis = 1).rename(columns={'level_0':'model'})
  {*parameters_dic.keys()}-{*parametros['model'].unique()}
  mg = df["name"].unique()
  elasticidades = set(mg.flatten())

  parameters_elasticidades_dic={x:parameters_dic[x] for x in elasticidades}
  parametros_elasticidades=pd.concat(parameters_elasticidades_dic).reset_index().drop('level_1', axis = 1).rename(columns={'level_0':'model'})
  elasticidades_cruzadas=parametros_elasticidades.pivot(index='model', columns='type', values='parameter').copy()
  elasticidades_cruzadas.drop(["kilo_lw","pandemic","t","const"], axis=1, inplace=True, errors='ignore')
  plt.subplots(figsize=(20,15))
  sns.heatmap(elasticidades_cruzadas, yticklabels=elasticidades_cruzadas.columns, xticklabels=elasticidades_cruzadas.columns, 
              cmap="Spectral" ,annot=True)
  elasticidades_cruzadas.to_csv('elasticidades_condensada.csv')
#############------------------------MAIN----------------------------------#######
if __name__=="__main__":
    ls_data=condensada_analysis().deploy()
    r=correlaciones(a).deploy_model()
    make_regression(elasticities(ls_data).deploy_elasticities())

 
    