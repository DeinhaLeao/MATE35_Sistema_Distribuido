from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ****************** LEITURA DOS DATASET ***********************
dfMortalidadeCID10Original = pd.read_csv("C:/Users/Deinha/PycharmProjects/DataScience/Preparation/Resource"
                                         "/Obitos pResidenc por Unidade da Federacao segundo Grupo CID-10.csv",
                                         sep=",", encoding='ISO-8859-1')

# ****************** PREPARAÇÃO ***********************

# Reindexar a 1ª coluna e apagar coluna "Grupo CID-10"
dfMortalidadeCID10 = dfMortalidadeCID10Original
dfMortalidadeCID10 = dfMortalidadeCID10.set_index(dfMortalidadeCID10Original["Grupo CID-10"])
dfMortalidadeCID10 = dfMortalidadeCID10.drop(labels=["Grupo CID-10"], axis=1, inplace=False)


def converterParaFloat(valor):
    # Trocar os valores "-" por Nan e converte os valores para float para posteriomente utilizar os métodos do pandas
    valor_str = str(valor)
    result = valor

    if valor_str == "-":
        result = np.nan

    return float(result)


# Altarar os campos vazios/não preenchidos para Nan (neste dataset é vazio/não preenchidos representado por "-")
dfMortalidadeCID10 = dfMortalidadeCID10.applymap(lambda x: converterParaFloat(x))

# Excluir a observação(linha), o atributo(coluna) com a totalização
dfMortalidadeCID10 = dfMortalidadeCID10.drop(labels=["Total"], axis=0, inplace=False)
dfMortalidadeCID10 = dfMortalidadeCID10.drop(labels=["Total"], axis=1, inplace=False)

# Excluir observações (linhas) e atributos (colunas) 100% nulas, se houver
dfMortalidadeCID10.dropna(axis=0, how="all")
dfMortalidadeCID10.dropna(axis=1, how="all")

# ****************** FIM PREPARAÇÃO ***********************

# ****************** ANÁLISE DOS DADOS ***********************
# Análise da Distribuição
print("-=--=-=--=-= Qtde de linhas e colunas do Dataset Mortalidade por Unidade da Federeal por Grupo CID-10 -=-=--==")
print(dfMortalidadeCID10.shape, "\n")

print("-=--=-=--=-=Tipos de Dados do Dataset Mortalidade por Unidade da Federeal por Grupo CID-10 -=-=--=-=--=-=-=-==")
dfMortalidadeCID10.info()
print("\n")

print("-=--=-=--=-=-=-=-=- Amostra dos Campos da Mortalidade por Unidade da Federeal por Grupo CID-10 -=-=--=-==-=-=-")
print(dfMortalidadeCID10.head(5))
print(dfMortalidadeCID10.tail(5), "\n")

print("-=--=-=-=-=-=-=-=-=- Distribuição da Mortalidade por Unidade da Federeal por Groupo CID-10 -=-=--=-==-=-=--=-=")
# Exploração dos dados para os valores numéricos
dfResumoMortalidadeCID10 = dfMortalidadeCID10.describe()
print(dfResumoMortalidadeCID10, "\n")

# ****************** FIM ANÁLISE DOS DADOS ***********************

# ****************** IMPUTAÇÃO ***********************
# Trabalho Prático Parte 2
# Analisar dados faltantes.

print("-=-=--=-=--=-=-=-=-=- Dados Faltantes da Mortalidade por Unidade da Federeal por Groupo CID-10 =-=--=-=-=-=-=-", '\n')
# Trocar NaN por 0
dfMortalidadeCID10FaltantesZerados = dfMortalidadeCID10
dfMortalidadeCID10FaltantesZerados = dfMortalidadeCID10FaltantesZerados.fillna(0)
print("=-=--=-= CID-10 - Distribuição dos Dados Faltantes Zerados =-=--=-=")
print(dfMortalidadeCID10FaltantesZerados.describe(), '\n')

# Inverter a linha e a coluna da tabela para calcular as médias/mediana pelas observações
dfMortalidadeCID10FaltantesTransposta = dfMortalidadeCID10
dfMortalidadeCID10FaltantesTransposta.transpose()

# Trocar Nan pela média das observações
dfMortalidadeCID10FaltantesMedia = dfMortalidadeCID10FaltantesTransposta
dfMortalidadeCID10FaltantesMedia = dfMortalidadeCID10FaltantesMedia.fillna(dfMortalidadeCID10FaltantesMedia.mean())
print("=-=--=-= CID-10 - Distribuição dos Dados Faltantes com a Média =-=--=-=")
print(dfMortalidadeCID10FaltantesMedia.describe(), '\n')

# Trocar Nan pela mediana das observações
dfMortalidadeCID10FaltantesMediana = dfMortalidadeCID10FaltantesTransposta
dfMortalidadeCID10FaltantesMediana = dfMortalidadeCID10FaltantesMediana.fillna(dfMortalidadeCID10FaltantesMediana.median())
print("=-=--=-= CID-10 - Distribuição dos Dados Faltantes Mediana =-=--=-=")
print(dfMortalidadeCID10FaltantesMediana.describe(), '\n')

# ****************** FIM IMPUTAÇÃO ***********************

# ****************** LINKAGE ***********************
# Etapa 2
# ****************** FIM LINKAGE ***********************

# ****************** VSUALIZAÇÃO ***********************

plt.close('all')

print("-=-=--=-=--=-=-=-=-=- Gráfico da Mortalidade da Bahia por Groupo CID-10 -=-=--=-=--=-=-=-=-=-")
plt.figure(1)
plt.subplot(111)
plt.plot(dfMortalidadeCID10FaltantesMedia.index, dfMortalidadeCID10FaltantesMedia['BA'], "ro")
plt.xlabel("Doenças por CID-10")
plt.ylabel("Total de Mortes")
plt.xticks(rotation=90)
plt.title('Grupo CID-10')
#plt.show()

print("-=-=--=-=--=-=-=-=-=- Histograma da Mortalidade da Bahia por Groupo CID-10 -=-=--=-=--=-=-=-=-=-")
plt.figure(2)
plt.subplot(211)
plt.xlabel("Doenças por CID-10")
plt.ylabel("Total de Mortes")
plt.xticks(rotation=90)
plt.hist(dfMortalidadeCID10FaltantesMedia["BA"] )
#plt.show()

print("-=-=--=-=--=-=-=-=-=- Heatmaps da Mortalidade por Unidade da Federeal por Grupo CID-10 -=-=--=-=--=-=-=-=-=-")
sns.set()
plt.figure(3)
plt.subplot(311)
#columns = ["RO", "AC",	"AM", "RR", "PA", "AP", "TO", "MA", "PI", "CE", "RN", "PB",
#                                            "PE", "AL", "SE", "BA", "MG", "ES", "RJ", "SP", "PR", "SC", "RS", "MS",
#                                            "MT", "GO", "DF"]

dfMortalidadeCID10FaltantesMedia.pivot_table(index="Grupo CID-10")
sns.heatmap(dfMortalidadeCID10FaltantesMedia, annot=True, fmt="g", cmap="viridis", robust=True, vmin=1, vmax=1000000,
            center=0, square=True, )
plt.show()

# ****************** FIM VSUALIZAÇÃO ***********************
