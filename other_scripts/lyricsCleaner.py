# - csv header: artist,title,album,decade,year,lyric,negative,neutral,positive,compound
import csv

chaves = ["(live", "live)", "[live", "live]", "live at", "version", "acoustic", "demo", "mix", "remix", "remastered ", "greatest hits ", "rock in rio"]

def paraRemover(row):
	for chave in chaves:
		if chave in row[1].lower() or chave in row[2].lower():
			return 1
	return 0

def colocaAspas(row): #coloca cada elemento da linha entre aspas devido aos elementos que tem virgula nao se perderem no csv
	for i in range(len(row)):
		row[i] = '''"'''+row[i]+'''"'''


with open('lyrics_analysis.csv', 'r', encoding="utf8") as file:
	reader = csv.reader(file)
	arq_clean = open("lyrics_analysis_Clean.csv", "w", encoding="utf8")
	arq2_leftover = open("lyrics_analysis_Leftover.csv", "w", encoding="utf8")
	for row in reader:
		colocaAspas(row)
		if(paraRemover(row)):
			arq2_leftover.write(",".join(row)+"\n")
		else:
			arq_clean.write(",".join(row)+"\n")
	arq_clean.close()
	arq2_leftover.close()
