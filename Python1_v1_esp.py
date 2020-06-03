###############3 Python


######  ScriptV1
####### Configuraciones iniciales de la sesión de Spark
##./bin/pyspark

from pyspark.sql import SparkSession
import os

MAX_MEMORY = "8g"
spark = SparkSession.builder \
					.master("local")\
                    .appName('TesisV1')\
                    .config("spark.executor.memory", MAX_MEMORY) \
                    .config("spark.driver.memory", MAX_MEMORY) \
                    .getOrCreate()



################# Creación de dataframes
facebook_df = spark.read.format('json').load("/home/ml/tesis/archivo8.json")
facebook_df = facebook_df.select("text")
twitter_df = spark.read.format('json').load("/home/ml/tesis/archivo7.json")
twitter_df = twitter_df.select("text")
redesSociales_df = facebook_df.union(twitter_df)
redesSociales_df.count()
redesSociales_df = redesSociales_df.dropDuplicates()
redesSociales_df.select("text").show(5,False)
################ Limpieza de textoo
from pyspark.sql.functions import *
def removerUsuarios(column):
	return trim(lower(regexp_replace(column,'@([^ ]+)', ''))).alias('stopped')


redesSociales_df=redesSociales_df.withColumn('text',removerUsuarios('text'))



def removerHashtag(column):
	return trim(lower(regexp_replace(column,'#([^ ]+)', ''))).alias('stopped')


redesSociales_df=redesSociales_df.withColumn('text',removerHashtag('text'))




		# test sin modificar el dataframe
#idioma_df.select(removerUsuarios(col('text'))).show(truncate=False)


####### remover direcciones http
def removerDirecciones(column):
	return trim(lower(regexp_replace(column,'https?://([^ ]+)', ''))).alias('stopped')


redesSociales_df=redesSociales_df.withColumn('text',removerDirecciones('text'))
	
		#test sin modificar dataframee
#idioma_df.select(removerDirecciones(col('text'))).show(truncate=False)


######## Remover puntuación y números

     ##return trim(lower(regexp_replace(column,'[^\sa-zA-Z]', ''))).alias('stopped')
###### funciona,deja incljusive tildes
def removerPuntuacionNumeros(column):
     return trim(lower(regexp_replace(column,'[^\s\p{L}]', ''))).alias('stopped')


redesSociales_df=redesSociales_df.withColumn('text',removerPuntuacionNumeros('text'))

def funcionEliminarSlash(column):
	return trim(lower(regexp_replace(column,'\\n', ''))).alias('stopped')



redesSociales_df=redesSociales_df.withColumn('text',funcionEliminarSlash('text'))


redesSociales_df = redesSociales_df.select("text")
redesSociales_df = redesSociales_df.na.drop()

###### Archivo de python clasificador_idioma.py

#import fasttext
#model = fasttext.load_model("deteccion_idioma.bin")
#def predecir_idioma(msg):
#    pred = model.predict(msg)[0][0]
#    return pred


	

###### llamada al archivo externo de clasificación de idioma

from pyspark.sql.functions import col, udf
sc.addFile('/home/ml/Downloads/spark-2.4.5-bin-hadoop2.7/deteccion_idioma.bin')
sc.addPyFile('/home/ml/Downloads/spark-2.4.5-bin-hadoop2.7/clasificador_idioma.py')
import clasificador_idioma
udf_predecir_idioma = udf(clasificador_idioma.predecir_idioma)

redesSociales_df = redesSociales_df.withColumn('idioma_predicho',
                                   udf_predecir_idioma(col('text')))


##### Para dejar solo código de idioma
def removerCodigo(column):
	return trim(lower(regexp_replace(column,'__label__', ''))).alias('stopped')


redesSociales_df=redesSociales_df.withColumn('idioma_predicho',removerCodigo('idioma_predicho'))								   								   
redesSociales_df=redesSociales_df.where(redesSociales_df.idioma_predicho == "spa")



################################ Fase 2: Modelo de detección de sentimiento
########### Creación de data frames
from pyspark.sql.functions import *
aprendizajemaquina_df1 = spark.read.text("/home/ml/Downloads/TwitterSentimentDataset-master/tweets_pos_clean.txt")
aprendizajemaquina_df1 = aprendizajemaquina_df1.withColumn("categoria",lit("bueno"))
aprendizajemaquina_df1.count()
aprendizajemaquina_df2 = spark.read.text("/home/ml/Downloads/TwitterSentimentDataset-master/tweets_neg_clean.txt")
aprendizajemaquina_df2 = aprendizajemaquina_df2.withColumn("categoria",lit("malo"))
aprendizajemaquina_df2.count()
aprendizajemaquina_df = aprendizajemaquina_df1.union(aprendizajemaquina_df2)
aprendizajemaquina_df.count()
aprendizajemaquina_df.show(2)
aprendizajemaquina_df.orderBy(rand()).show(5,False)



#################limpieza del textoo


######## remover @ nombres de usuarios
def removerUsuarios(column):
	return trim(lower(regexp_replace(column,'@([^ ]+)', ''))).alias('stopped')


aprendizajemaquina_df=aprendizajemaquina_df.withColumn('value',removerUsuarios('value'))

		# test sin modificar el dataframe
#aprendizajemaquina_df.select(removerUsuarios(col('value'))).show(truncate=False)


####### remover direcciones http
def removerDirecciones(column):
	return trim(lower(regexp_replace(column,'https?://([^ ]+)', ''))).alias('stopped')


aprendizajemaquina_df=aprendizajemaquina_df.withColumn('value',removerDirecciones('value'))
	
		#test sin modificar dataframee
#aprendizajemaquina_df.select(removerDirecciones(col('value'))).show(truncate=False)


######## Remover puntuación y números

     ##return trim(lower(regexp_replace(column,'[^\sa-zA-Z]', ''))).alias('stopped')
###### funciona,deja incljusive tildes
def removerPuntuacionNumeros(column):
     return trim(lower(regexp_replace(column,'[^\s\p{L}]', ''))).alias('stopped')


aprendizajemaquina_df=aprendizajemaquina_df.withColumn('value',removerPuntuacionNumeros('value'))

	#test sin modificar dataframe
#aprendizajemaquina_df.select(removerPuntuacionNumeros(col('value'))).show(truncate=False)

#####, RegexTokenizer
######### tokenizer
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
tokenizer = Tokenizer(inputCol="value", outputCol="palabras")
aprendizajemaquina_df = tokenizer.transform(aprendizajemaquina_df)
aprendizajemaquina_df.select("value", "palabras").show(6,False)
aprendizajemaquina_df.select("palabras").show(5,False)



######### stopwords
from pyspark.ml.feature import StopWordsRemover


listaConectores= StopWordsRemover.loadDefaultStopWords("spanish")
remover = StopWordsRemover(inputCol="palabras", outputCol="filtro_conectores", stopWords=listaConectores)
aprendizajemaquina_df = remover.transform(aprendizajemaquina_df)
aprendizajemaquina_df.select("palabras").show(5,False)
aprendizajemaquina_df.select("filtro_conectores").show(5,False)


############# Entrenamiento del modelo 
###### Palabras a vectores Word2Vec

from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline

w2v = Word2Vec(vectorSize=100, minCount=0, inputCol="filtro_conectores", outputCol="vectores")
aprendizajemaquina_word2vec_modelo = w2v.fit(aprendizajemaquina_df)
aprendizajemaquina_df = aprendizajemaquina_word2vec_modelo.transform(aprendizajemaquina_df)
aprendizajemaquina_df.select("categoria","filtro_conectores","vectores").orderBy(rand()).show(5)

############### Repartición de datos
aprendizajemaquina_entrenamiento_df, aprendizajemaquina_evaluacion_df = aprendizajemaquina_df.randomSplit([0.8, 0.2])



################ Regresión logística

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

si = StringIndexer(inputCol="categoria", outputCol="etiquetas_ml")
clasificador_rl = LogisticRegression(family="multinomial", featuresCol="vectores"\
, labelCol="etiquetas_ml",predictionCol="prediccion",probabilityCol="probabilidad")   
clasificador_rl_pipeline = Pipeline(stages=[si,clasificador_rl])




########################################################## FASE 3: 

#####, RegexTokenizer
######### tokenizer
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
tokenizer = Tokenizer(inputCol="text", outputCol="palabras")
redesSociales_df = tokenizer.transform(redesSociales_df)
redesSociales_df.select("text", "palabras").show(6,False)
redesSociales_df.select("text").show(5,False)



######### stopwords
from pyspark.ml.feature import StopWordsRemover


listaConectores= StopWordsRemover.loadDefaultStopWords("spanish")
remover = StopWordsRemover(inputCol="palabras", outputCol="filtro_conectores", stopWords=listaConectores)
redesSociales_df = remover.transform(redesSociales_df)
redesSociales_df.select("palabras").show(5,False)
redesSociales_df.select("filtro_conectores").show(5,False)


############# Entrenamiento del modelo 
###### Palabras a vectores Word2Vec

from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline

w2v = Word2Vec(vectorSize=100, minCount=0, inputCol="filtro_conectores", outputCol="vectores")
redesSociales_word2vec_modelo = w2v.fit(redesSociales_df)
redesSociales_df = redesSociales_word2vec_modelo.transform(redesSociales_df)
redesSociales_df.select("filtro_conectores","vectores").orderBy(rand()).show(5)







modelo_sentimiento = clasificador_rl_pipeline.fit(aprendizajemaquina_entrenamiento_df)
prediccion_sentimiento_redesSociales_df = modelo_sentimiento.transform(redesSociales_df)


def prediccionLiteral(column):
	if column == 1.0:
		return "Bueno"
	else:
		return "Malo"



prediccionLiteral_udf = udf(prediccionLiteral)
prediccion_sentimiento_redesSociales_df=prediccion_sentimiento_redesSociales_df\
	.withColumn('sentimiento',prediccionLiteral_udf(prediccion_sentimiento_redesSociales_df.prediccion))

prediccion_sentimiento_redesSociales_df.select("text", "idioma_predicho", "sentimiento").show()


buenos = prediccion_sentimiento_redesSociales_df\
	.where(prediccion_sentimiento_redesSociales_df.sentimiento == "Bueno").count()
malos = prediccion_sentimiento_redesSociales_df\
	.where(prediccion_sentimiento_redesSociales_df.sentimiento == "Malo").count()
total =prediccion_sentimiento_redesSociales_df.count()
porcentajebuenos = int(buenos * 100.00 /total)
porcentajemalos = 100 -porcentajebuenos
print("Buenos = " + str(porcentajebuenos) + "%"\
	, "Malos = "+str(porcentajemalos)+"%", "Total de comentarios ="+str(total))

prediccion_sentimiento_redesSociales_df.where(prediccion_sentimiento_redesSociales_df.sentimiento == "Malo").select("text").show()


