# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

#importar librerias
import pyodbc
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing


#crear conexion############################################
conn=pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                    'SERVER=******;'
                    'DATABASE=****;'
                    'Trusted_Connection=yes;')

"""
En caso de hacer este código cómo servicio web, necesitaria ingestar las tablas siguientes de la base de datos 
y obtener el id de la empresa que hace la cosnulta y el id_perfil que busca.
El resultado se almacena en una dataframe creado al final.

"""
# INPUTS#######

id_empresa=10
id_perfil=1
id_subperfil=1



#############IMPORTAR TABLAS###########################
#obtener encuesta
encuesta = pd.read_sql("select * from ****.dbo.MLVariablesEmpresa", conn)

#obtener data frame variables empresa
variables_emp = pd.read_sql("select * from ****.dbo.empresas", conn)

#obtener data frame variables relación empresas
variables_rel = pd.read_sql("select * from ****.dbo.relacionEmpresa", conn)

#tabla relación id perfil y su descripción
id_perf_desc = pd.read_sql("select * from ****.dbo.perfiles", conn)

#tabla relacion idperfiles idempresa
idperf_idempresa = pd.read_sql("select * from *****.perfilesEmpresa", conn)

#tabla descripción tipos de perfil
idperf_type = pd.read_sql("select * from ****.dbo.tiposPerfil", conn)

#tabla variables con id variables
idvar_var = pd.read_sql("select * from ****.dbo.MLvariables", conn)

#######################################################


#rellenar dataframe variables rel (relación empresa)###################################

#eliminar las columnas que no necesito
variables_rel=variables_rel.drop(columns=['activo','estado','ultimaFecha'])

#rellenar valores de las columnas (paso eliminable en producción)##########
#import numpy as np

data_relcomant = np.random.randint(0,10,size=6)

data_valorig= np.random.randint(1,5,size=6)

data_valdest= np.random.randint(1,5,size=6)

data_cumpfso= np.random.randint(0,100,size=6)

data_cumpfsd= np.random.randint(0,100,size=6)

data_cumpcalor= np.random.randint(0,100,size=6)

data_cumpcaldes= np.random.randint(0,100,size=6)

data_valco= np.random.randint(1,4,size=6)

data_valcd= np.random.randint(1,4,size=6)

data_anor= np.random.randint(0,10,size=6)

data_andes= np.random.randint(0,10,size=6)

data_soldpo= np.random.randint(0,10,size=6)

data_soldpd= np.random.randint(0,10,size=6)

        
lista=variables_rel.columns

for columna in lista:
    print('variables_rel[]=') 
        
variables_rel['mlRelComAnt']=data_relcomant
variables_rel['mlValoracionOrigen']=data_valorig
variables_rel['mlValoracionDestino']=data_valdest
variables_rel['mlCumplFecSerOrigen']=data_cumpfso
variables_rel['mlCumplFecSerDestino']=data_cumpfsd
variables_rel['mlCumplCalidadOrigen']=data_cumpcalor
variables_rel['mlCumplCalidadDestino']=data_cumpcaldes
variables_rel['mlValoracionCosteOrigen']=data_valco
variables_rel['mlValoracionCosteDestino']=data_valcd
variables_rel['mlAnulacionesOrigen']=data_anor
variables_rel['mlAnulacionesDestino']=data_andes
variables_rel['mlSolicitudDemoraPedOrigen']=data_soldpo
variables_rel['mlSolicitudDemoraPedDestino']=data_soldpd
###########eliminable en producción############

#seleccionar de encuesta las filas con el id de la empresa y el id perfil que busca
encuesta=encuesta.loc[(encuesta['id_empresa'] == id_empresa) & (encuesta['id_perfil'] == id_perfil )]
#encuesta=encuesta.loc[(encuesta['id_empresa'] == 10)] #obtener la encuesta de la empresa 10

#########RELLENADAR DF VARIABLES EMPRESA################

#eliminar columnas 
variables_emp=variables_emp.drop(columns=['razonSocial','nombreComercial','cif','domicilio',
                                          'domicilio2','codigoPostal','poblacion','codigoPais','provincia','telefono1',
                                          'telefono2','movil','email','web','registroMercantil','logoEmpresa','fechaAlta',
                                          'activo','ambito','orden','estado','apiKey','servidor','baseDatos',
                                          'usuarioSQL','nci','cuentaBC','pkBC'])


#habrá que hacer que en las columns TRAZABILIDAD,MLCERTIFCALIDAD,MLLEANMANUFACTURING haya que hacer que false sea 0 y true 1
#########eliminable en producción########
#datos tabla var empresas

data_nivex = np.random.randint(1,5,size=28)

data_valorcost= np.random.randint(1,4,size=28)

data_dimen= np.random.randint(1,200,size=28)

data_fechalta= np.random.randint(1800,2020,size=28)

data_trazab= np.random.randint(0,1,size=28)

data_digit= np.random.randint(1,3,size=28)

data_pagaplaz= np.random.randint(1,5,size=28)

data_reconomarc= np.random.randint(1,3,size=28)

data_certcal= np.random.randint(0,1,size=28)

data_leanmanu= np.random.randint(0,1,size=28)


lista=variables_emp.columns
print(lista)


variables_emp['mlNivelExigencia']=data_nivex
variables_emp['mlValoracionCoste']=data_valorcost
variables_emp['mlDimension']=data_dimen
variables_emp['mlFechaAlta']=data_fechalta
variables_emp['mlTrazabilidad']=data_trazab
variables_emp['mlDigitalizacion']=data_digit
variables_emp['mlPagaPlazos']=data_pagaplaz
variables_emp['mlReconocimientoMarca']=data_reconomarc
variables_emp['mlCertificadosCalidad']=data_certcal
variables_emp['mlLeanManufacturing']=data_leanmanu
#########eliminable en producción####################

if (id_empresa in variales_rel['idEmpresaOrigen']= True):
    varialbes_rel=variables_rel.drop(columns=['mlValoracionDestino','mlCumplFecSerDestino','mlCumplCalidadDestino,
                                              'mlValoracionCosteDestino','mlAnulacionesDestino','mlSolicitudDemoraPedDestino'])
    elif (id_empresa in variales_rel['idEmpresaDestino']= True):
                                              

#merge de variables emp y var rel by id 
df_merged=pd.merge(variables_emp, variables_rel, left_on='id', right_on='idEmpresaDestino', how='left')

#drop rows with nans and elimnate reundand id columns
df_merged_nonan = df_merged.dropna()
df_merged_clean=df_merged_nonan.drop(columns=['idEmpresaDestino','idEmpresaOrigen'])

#set id as index
df_merged_clean_index=df_merged_clean.set_index('id')



###################sustituir variables por id variables

idvar_var_2=idvar_var.drop(columns=['descripcion'])

#change column name id for id_var
idvar_var_2=idvar_var_2.rename(columns={'id':'id_var'})

#pasar columnas a filas de variables, merge por nombre variable creando id_var, poner id var como header
list_var=list(df_merged_clean_index.columns.values)

#cambiar nombres variables por sus indices
df_merged_idvar = df_merged_clean_index.rename(columns = {'mlNivelExigencia':'1', 'mlValoracionCoste':'2', 'mlDimension':'3',
                                        'mlFechaAlta':'4', 'mlTrazabilidad':'5', 'mlDigitalizacion':'6', 'mlPagaPlazos':'7',
                                        'mlReconocimientoMarca':'8', 'mlCertificadosCalidad':'9', 'mlLeanManufacturing':'10',
                                         'mlRelComAnt':'11', 'mlValoracionOrigen':'12',
                                        'mlValoracionDestino':'13', 'mlCumplFecSerOrigen':'14', 'mlCumplFecSerDestino':'15',
                                        'mlCumplCalidadOrigen':'16', 'mlCumplCalidadDestino':'17', 'mlValoracionCosteOrigen':'18',
                                        'mlValoracionCosteDestino':'19', 'mlAnulacionesOrigen':'20', 'mlAnulacionesDestino':'21',
                                        'mlSolicitudDemoraPedOrigen':'22', 'mlSolicitudDemoraPedDestino':'23'}, inplace = False)



#####################################MODIFICACIONES PARA CALCULO VARIABLES###########

#transformar la variable mlFechaAlta para que cuanto más anntigua mejor
df_merged_idvar['4']=3000 - df_merged_idvar['4']

#♥extract columns from df idor y iddest
#df_rel_ids=df_merged_idvar[['id','idEmpresaOrigen','idEmpresaDestino']]

#drop rows with nans from df_rel_ids
#df_rel_ids_nonan = df_rel_ids.dropna()


#df_no_ids=df_merged_idvar.drop(columns=['id','idEmpresaOrigen','idEmpresaDestino'])

#drop rows with nans from df_no_ids
#df_no_ids_nonan = df_no_ids.dropna()

#normalizar df_no_ids

#from sklearn import preprocessing

column_maxes = df_merged_idvar.max()
df_max = column_maxes.max()
column_mins = df_merged_idvar.min()
df_min = column_mins.min()
normalized_df = (df_merged_idvar - df_min) / (df_max - df_min)

###############MODIFICACIÓN PARA CALCULO ENCUESTA###################################


#transformar datos de la encuesta para que el 1 sea el mejor y el 10 el peor
df_encuesta_trans=1-(encuesta['orden']/23)+1


#transponer df_var_normalized para poder calcular el resultado de forma matricial
df_var_transposed = normalized_df.T


#pasar a numpy para multiplicar de forma matricial
import numpy as np

matriz_var=np.array(df_var_transposed)

matriz_enc=np.array(df_encuesta_trans)


#multiplicación
result=np.dot(matriz_enc,matriz_var)





#crear data frame con el resultado y las empresas
df_final = pd.DataFrame(result, columns = [
    "resultado"])

df_final=df_final.set_index(df_merged_clean['id'])

#crear data frame resultado
df_for_tipo_id=pd.merge(idperf_idempresa, df_merged_clean, left_on='idEmpresa', right_on='id', how='left')
df_for_tipo_id = df_for_tipo_id[['id','idTipoPerfil']]

id_empresa=encuesta['id_empresa']
id_perfil=encuesta['id_perfil']
id_tipo=df_for_tipo_id['idTipoPerfil']
id_empresa_dest=df_merged_clean['id']
resultado=df_final["resultado"]


df = pd.DataFrame(list(zip(id_empresa, id_perfil,id_tipo,id_empresa_dest,resultado)),
               columns =['id_empresa', 'id_perfil','id_tipo','id_empresa_dest','resultado'])

#Antes de insertar los resulatdos, vacir la tabla
#Insertar resultados en tabla de SQL server (resultados)

cursor = conn.cursor()

sqlcommand=("INSERT INTO ****.dbo.MLResultados([id_empresa],[id_perfil],[id_tipo],[id_empresa_dest],[resultado]) VALUES (?,?,?,?,?)")

#values=ranking.index.tolist()


for index, row in df.iterrows():
    cursor.execute(sqlcommand,row['id_empresa'],
                   row['id_perfil'],
                   row['id_tipo'],
                   row['id_empresa_dest'],
                   row['resultado'])

conn.commit()    

#cursor.close()
#conn.close()




