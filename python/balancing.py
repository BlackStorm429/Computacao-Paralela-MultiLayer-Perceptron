import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Carregar o arquivo .csv
df = pd.read_csv('diabetes.csv')

# Separar as features e o alvo (última coluna - diabetes)
X = df.drop('diabetes', axis=1)  # ou o nome da sua coluna
y = df['diabetes']

# Verificar a distribuição antes do balanceamento
print(f'Distribuição antes do balanceamento:\n{y.value_counts()}')

# Aqui você pode escolher entre undersampling ou oversampling:
# Caso escolha oversampling:
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Caso escolha undersampling (comente o código acima e descomente o abaixo):
# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X, y)

# Verificar a distribuição após o balanceamento
print(f'Distribuição após o balanceamento:\n{y_resampled.value_counts()}')

# Criar o novo dataframe balanceado
df_balanced = pd.concat([X_resampled, y_resampled], axis=1)

# Salvar o novo .csv balanceado
df_balanced.to_csv('diabetes_balanced.csv', index=False)
