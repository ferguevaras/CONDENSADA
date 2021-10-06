import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import datetime
import os
import seaborn as sns
import plotly.offline as pyoff
import plotly.graph_objs as go
class GAP():
  
  def __init__(self):
    #Setting default vars for our analysis
    self.path = '/content/drive/MyDrive/Condensada/Raw_data/LECHE_CONDENSADA.csv'
    self.mercado = 'TOTAL AUTOS SCANNING MEXICO'
    self.year =2020
    self.df=''
    self.tracklist=['NESTLE LA LECHERA LECHE CONDENSADA LATA 387GR NAL 7501058617873','PRONTO LECHE CONDENSADA LATA 380GRS 1PZA 7501200486005','CONTROLLED LABEL']

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
    self.df=self.df[(self.df['MERCADO']==self.mercado) & (self.df['ANIO']>=self.year)]
    self.transform()
    self.df=self.df[self.df['DESCRIPCION'].isin(self.tracklist)]
    self.assign_date()
    self.df=self.df[['date','DESCRIPCION','ITEM','VentasValor','VentasUnidades','precio','SEM']]
    self.df=self.df[self.df['date']>'2020-08-31']

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
  #Take the control of the deployment %MLOPS
  def deploy_gap(self):
    self.read_file_validation()
    self.wrangling()
    return self.df
#Function that use fbprophet in order to forecast Condensada with extra regressors
def create_series(df_model):
  #Create Lechera DF
  df_lech=df_model[df_model['DESCRIPCION']=='NESTLE LA LECHERA LECHE CONDENSADA LATA 387GR NAL 7501058617873'][['date','VentasValor','precio']]
  df_lech.columns=['ds','y','price_lechera']
  #Create pronto DF
  df_pronto=df_model[df_model['DESCRIPCION']=='PRONTO LECHE CONDENSADA LATA 380GRS 1PZA 7501200486005']
  df_pronto=df_pronto[['date','VentasValor','precio']]
  df_pronto.columns=['ds','ventas_pronto','precio_pronto']
  #First MERGE to join both series
  df_merged1=df_pronto.merge(df_lech,on='ds',how='right')
  #Create Controlled label DF
  df_cb=df_model[df_model['DESCRIPCION']=='CONTROLLED LABEL'][['date','VentasValor','precio']]
  df_cb.columns=['ds','ventas_controlledlabel','precio_controlledlabel']
  #Second Merge to join the last  serie
  df_merged2=df_merged1.merge(df_cb,on='ds',how='left')
  #GAP FORMULAS
  df_merged2['gap_pronto']=(df_merged2['price_lechera']/df_merged2['precio_pronto'])-1
  df_merged2['gap_controlled']=(df_merged2['price_lechera']/df_merged2['precio_controlledlabel'])-1
  df_merged2.drop(['precio_controlledlabel','price_lechera','precio_pronto'],axis=1,inplace=True)
  df_model=df_merged2[:-8]
  df_model=df_model[df_model['ds']>='2021-01-04']
  gaps=df_model[['ds','gap_pronto','gap_controlled']]
  #Prepare prophet columns names for the model
  dsy=df_model[['ds','y']]
  #Config prophet metrics to create the forecast
  m = Prophet(yearly_seasonality=3,weekly_seasonality=18)
  #Adding extra regressors
  m.add_regressor('gap_pronto')
  m.add_regressor('gap_controlled')
  m.fit(df_model)
  future = m.make_future_dataframe(periods=7, freq='7D')
  future = pd.merge(future, gaps,how='left',on='ds')
  #Fill the regressors series. Keeping default gap value
  future['gap_pronto']= np.where(future.gap_pronto.isnull(), 0.243151,future['gap_pronto'])
  future['gap_controlled']= np.where(future.gap_controlled.isnull(),0.143074,future['gap_controlled'])
  forecast = m.predict(future)
  aux = df_model[['ds','y']]
  aux.columns=['ds','yhat']
  #Create Final Datas
  dfinal = pd.concat([aux,forecast[['ds','yhat','yhat_lower', 'yhat_upper']].tail(7)], ignore_index=True)
  f = m.plot_components(forecast)
  fig = go.Figure()
  # Create and style traces
  fig.add_trace(go.Scatter(x=df_model['ds'], y=df_model['y'], name='La Lechera',))
  fig.add_trace(go.Scatter(x=forecast['ds'].tail(8), y=forecast['yhat'].tail(7),name='Predicted- GAP 0.243151', opacity=1,))
  fig.show()
######------------MAIN-------------#####
if __name__=="__main__":
  create_series(GAP().deploy_gap())
