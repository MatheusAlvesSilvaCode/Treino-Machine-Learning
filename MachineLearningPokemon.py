import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Carregar os dados (com o separador ';' se necessário)
data = pd.read_csv(r'C:\Users\mathe\OneDrive\Área de Trabalho\Estágio\Machine Learning Estudo\DataSet-PokemonForMachineLearning.csv', sep=';')

# Exibir os dados para checar a leitura
#print(data.head(5))  # Exibe as primeiras 5 linhas do DataFrame

# Verifique as colunas do seu DataFrame para garantir que foram lidas corretamente
#print(data.columns)

data[data.isnull().any(axis=1)] # Mostre todas as linhas da tabela que tenha pelo menos um valor nulo 

before_row = data.shape[0]

data = data.dropna()

after_row = data.shape[0]

#result_row = before_row - after_row # Apenas uma linha foi retirada 'Saída 1'

clean_data = data.copy() # Copiando data em uma variavel para manter o data original.
# 'Binarizando' coluna.

# Esse valor é apenas para estudo e não realmente mmostra todos os Pokémons lendários!! 
clean_data['Legendary'] = (clean_data['Total'] > 399) *1 # Criando coluna 'Legendary' onde se o total for acima de 400 true. 
#print(clean_data['Legendary']) 


y = clean_data [['Legendary']].copy() # Copiando coluna Legendary para variavel y  

clean_data['Total'].head(5) # Exibindo primeiros 5 resultados da coluna total.

pokemon_feature = ['HP','Attack','Defense','Speed','Special']

x = clean_data[pokemon_feature].copy() # Na variavel x, acessando essas colunas em clean_data

x.columns
y.columns

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)

pokemon_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0) 
pokemon_classifier.fit(x_train, y_train)

type(pokemon_classifier)

predicoes = pokemon_classifier.predict(x_test)
predicoes[:10]

y_test['Legendary'][:10] # Mostra toda a linha ate o 10
print(accuracy_score(y_true = y_test, y_pred = predicoes)) # 0.86 significa que a precisão está de 86%



# Junta os nomes dos Pokémon ao conjunto de teste
x_test_with_names = x_test.copy()
x_test_with_names['Pokemon'] = data.loc[x_test.index, 'Pokemon']

# Adiciona a predição (0 ou 1) à tabela
x_test_with_names['Predicted_Legendary'] = predicoes

# Mostra apenas os Pokémon que o modelo classificou como lendários
legendary_predicted = x_test_with_names[x_test_with_names['Predicted_Legendary'] == 1]
print(legendary_predicted[['Pokemon', 'Predicted_Legendary']])
print(len(legendary_predicted))
 


