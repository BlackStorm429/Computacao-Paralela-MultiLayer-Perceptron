def converte_string_para_numero(valor):
    """
    Função para converter strings específicas para valores numéricos.
    Mapeamento de strings:
    - Sexo:
      'Male' -> 1
      'Female' -> 0
    - Estado de saúde/informação:
      'No' -> 0
      'No Info' -> 0
      'Not Current' -> 0
      'Current' -> 1
      'Never' -> 0
    """
    if valor == "Male":
        return 1
    elif valor == "Female":
        return 0
    elif valor == "No" or valor == "No Info" or valor == "Not Current" or valor == "Never":
        return 0
    elif valor == "Current":
        return 1
    else:
        return 0  # Caso algum valor inesperado apareça

def processa_linha(linha):
    """
    Converte uma linha do arquivo 'diabetes_expanded.txt' para seus valores adequados.
    As colunas são:
    0: Sexo (Male/Female) -> 1 ou 0
    1-7: Atributos numéricos
    8: Classe (0 ou 1)
    """
    dados = linha.strip().split()  # Divide a linha em partes por espaços

    # Mapeamento de valores de string para números
    sexo = converte_string_para_numero(dados[0])  # Converte 'Male' ou 'Female' para 1 ou 0
    classe = int(dados[-1])  # Último valor é a classe (0 ou 1)

    # Os outros atributos são números e precisam ser convertidos
    atributos = [sexo] + [float(dado) if dado.replace('.', '', 1).isdigit() else 0 for dado in dados[1:-1]]

    return atributos, classe

def processa_arquivo(nome_arquivo):
    """
    Função para processar todas as linhas do arquivo 'diabetes_expanded.txt' e converter seus valores.
    """
    dataset = []  # Lista para armazenar os dados convertidos
    with open(nome_arquivo, 'r') as arquivo:
        for linha in arquivo:
            # Processa cada linha do arquivo
            atributos, classe = processa_linha(linha)
            dataset.append((atributos, classe))  # Adiciona a instância à lista
    
    return dataset

def salva_no_arquivo(dataset, nome_arquivo_saida):
    """
    Função para salvar os dados convertidos em um novo arquivo.
    """
    with open(nome_arquivo_saida, 'w') as arquivo:
        for atributos, classe in dataset:
            # Converte os atributos e a classe para uma linha de texto
            linha = ' '.join(map(str, atributos)) + ' ' + str(classe) + '\n'
            arquivo.write(linha)

# Função para testar com o arquivo
dataset = processa_arquivo("diabetes_balanced.txt")

# Salva os dados processados no novo arquivo
salva_no_arquivo(dataset, "diabetes_without_strings.txt")

print("Arquivo 'diabetes_without_strings.txt' criado com sucesso!")
