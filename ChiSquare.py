from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import VectorIndexer, ChiSqSelector		
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Varianles

Edad = 0
Genero = 1
Zona = 2
Fumador_Activo = 3
Diabetes = 4
ultimo_estado_de_Glicemia = 5
Enfermedad_Coronaria = 6
Tension_sistolica = 7
Tension_diastolica = 8
Colesterol_Total = 9
Trigliceridos = 10
Clasificacion_RCV_Global = 11
Glicemia_de_ayuno = 12
Perimetro_Abdominal = 13
Peso = 14
IMC = 15
CLAIFICACION_IMC = 16
Creatinina = 17
Factor_correccion = 18
Proteinuria = 19
Farmacos_Antihipertensivos = 20
Estatina = 21
Antidiabeticos = 22
Adherencia_tratamiento = 23

def leer_df():
	conf = SparkConf().setAppName("BaseDeDatos.csv").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)

	# Leemos el CSV
	rdd = sqlContext.read.csv("BaseDeDatos.csv", header=True).rdd

    # Filtrando datos vacios
	rdd = rdd.filter(
		lambda x: (		   x[Edad] != None and x[Genero] != '' and x[Zona] != None and x[Fumador_Activo] != None and x[Diabetes] != None and \
						   x[ultimo_estado_de_Glicemia] != None and x[Enfermedad_Coronaria] != None and x[Tension_sistolica] != None and x[Tension_diastolica] != None and \
						   x[Colesterol_Total] != None and x[Trigliceridos] != None and x[Clasificacion_RCV_Global] != None and x[Glicemia_de_ayuno] != None and x[Perimetro_Abdominal] != None and \
						   x[Peso] != None and x[IMC] != None and x[CLAIFICACION_IMC] != None and x[Creatinina] != None and x[Factor_correccion] != None and \
						   x[Proteinuria] != None and x[Farmacos_Antihipertensivos] != None and x[Estatina] != None and x[Antidiabeticos] != None and x[Adherencia_tratamiento] != None
                           ))

	rdd = rdd.map(
		lambda x: ( int(x[Edad]), int(x[Genero]), int(x[Zona]), int(x[Fumador_Activo]), int(x[Diabetes]), int(x[ultimo_estado_de_Glicemia]) ,
			int(x[Enfermedad_Coronaria]) , 
            int(x[Tension_sistolica]), 
            int(x[Tension_diastolica]), 
            int(x[Colesterol_Total]), 
            int(x[Trigliceridos]),
            float(x[Clasificacion_RCV_Global]), int(x[Glicemia_de_ayuno]), int(x[Perimetro_Abdominal]), int(x[Peso]), int(x[IMC]) , 
            int(x[CLAIFICACION_IMC]), int(x[Creatinina]),int(x[Factor_correccion]),
            int(x[Proteinuria]),int(x[Farmacos_Antihipertensivos]), int(x[Estatina]),int(x[Antidiabeticos]),int(x[Adherencia_tratamiento])
            ))

	df = rdd.toDF(["Edad","Genero","Zona","Fumador_Activo","Diabetes",
    "ultimo_estado_de_Glicemia","Enfermedad_Coronaria","Tension_sistolica","Tension_diastolica","Colesterol_Total",
    "Trigliceridos","Clasificacion_RCV_Global","Glicemia_de_ayuno","Perimetro_Abdominal","Peso",
    "IMC","CLAIFICACION_IMC","Creatinina","Factor_correccion","Proteinuria",
    "Farmacos_Antihipertensivos","Estatina","Antidiabeticos","Adherencia_tratamiento"])

	return df

def feature_selection(df):
	assembler = VectorAssembler(
		inputCols=["Edad","Genero","Zona","Fumador_Activo",
    "ultimo_estado_de_Glicemia","Enfermedad_Coronaria","Tension_sistolica",
    "Tension_diastolica","Colesterol_Total","Trigliceridos",
    "Clasificacion_RCV_Global","Glicemia_de_ayuno","Perimetro_Abdominal",
    "Peso","IMC","CLAIFICACION_IMC","Creatinina","Factor_correccion","Proteinuria",
    "Farmacos_Antihipertensivos","Estatina","Antidiabeticos","Adherencia_tratamiento"],
		outputCol="features")
	df = assembler.transform(df)

	indexer = VectorIndexer(
		inputCol="features", 
		outputCol="indexedFeatures",
		maxCategories=15)
	
	df = indexer.fit(df).transform(df)

	# Seleccionamos features que mas suman al modelo
	selector = ChiSqSelector(
		numTopFeatures=15,
		featuresCol="indexedFeatures",
		labelCol="Diabetes",
		outputCol="selectedFeatures")
	resultado = selector.fit(df).transform(df)
	resultado.select("features", "selectedFeatures").show(100)

def main():
	df = leer_df()
	feature_selection(df)
	#entrenamiento(df)
	
if __name__ == "__main__":
	main()
















