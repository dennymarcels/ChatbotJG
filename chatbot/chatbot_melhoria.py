# Construção do chatbot com Deep NLP

# Importação das bibliotecas
import numpy as np
import tensorflow as tf
import re
import time

# --- Parte 1 - Pré-processamento dos dados ---

# Importação das bases de dados
linhas = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversas = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Criação de um dicionário para mapear cada linha com seu ID
# Olá! - Olá!
# Tudo bem? - Tudo!
# Eu também!

# Olá! - Olá!
# Olá! - Tudo bem?
# Tudo bem? - Tudo
# Tudo - Eu também!

id_para_linha = {}
for linha in linhas:
    #print(linha)
    _linha = linha.split(' +++$+++ ')
    #print(_linha)
    if len(_linha) == 5:
        #print(_linha[4])
        id_para_linha[_linha[0]] = _linha[4]
        
# Criação de uma lista com todas as conversas
conversas_id = [] 
for conversa in conversas[:-1]:
    #print(conversa)
    _conversa = conversa.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    #print(_conversa)
    conversas_id.append(_conversa.split(','))
    
# Separação das perguntas e respostas
# 194 - 195 - 196 - 197

# 194 - 195
# 195 - 196
# 196 - 197
    
perguntas = []
respostas = []
for conversa in conversas_id:
    #print(conversa)
    #print('*****')
    for i in range(len(conversa) - 1):
        #print(i)
        perguntas.append(id_para_linha[conversa[i]])
        respostas.append(id_para_linha[conversa[i + 1]])

def limpa_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"i'm", "i am", texto)
    texto = re.sub(r"he's", "he is", texto)
    texto = re.sub(r"she's", "she is", texto)
    texto = re.sub(r"that's", "that is", texto)
    texto = re.sub(r"what's", "what is", texto)
    texto = re.sub(r"where's", "where is", texto)
    texto = re.sub(r"how's", "how is", texto)
    texto = re.sub(r"\'ll", " will", texto)
    texto = re.sub(r"\'ve", " have", texto)
    texto = re.sub(r"\'re", " are", texto)
    texto = re.sub(r"\'d", " would", texto)
    texto = re.sub(r"n't", " not", texto)
    texto = re.sub(r"won't", "will not", texto)
    texto = re.sub(r"can't", "cannot", texto)
    texto = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", texto)
    return texto    

limpa_texto("ExeMplo i'm #@")
            
# Limpeza das perguntas
perguntas_limpas = []
for pergunta in perguntas:
    perguntas_limpas.append(limpa_texto(pergunta))

# Limpeza das respostas
respostas_limpas = []
for resposta in respostas:
    respostas_limpas.append(limpa_texto(resposta))

# Retirada de perguntas e respostas que são muito curtas ou muito longas
perguntas_curtas = []
respostas_curtas = []
i = 0
for pergunta in perguntas_limpas:
    if 2 <= len(pergunta.split()) <= 25:
        perguntas_curtas.append(pergunta)
        respostas_curtas.append(respostas_limpas[i])
    i += 1
perguntas_limpas = []
respostas_limpas = []
i = 0
for resposta in respostas_curtas:
    if 2 <= len(resposta.split()) <= 25:
        respostas_limpas.append(resposta)
        perguntas_limpas.append(respostas_curtas[i])
    i += 1

# Criação de um dicionário que mapeia cada palavra e o número de ocorrências NLTK
palavras_contagem = {}
for pergunta in perguntas_limpas:
    #print(pergunta)    
    for palavra in pergunta.split():
        if palavra not in palavras_contagem:
            palavras_contagem[palavra] = 1
        else:
            palavras_contagem[palavra] += 1

for resposta in respostas_limpas:
    for palavra in resposta.split():
        if palavra not in palavras_contagem:
            palavras_contagem[palavra] = 1
        else:
            palavras_contagem[palavra] += 1

# Remoção de palavras não frequentes e tokenização (dois dicionários)
limite = 15
perguntas_palavras_int = {}
numero_palavra = 0
for palavra, contagem in palavras_contagem.items():
    #print(palavra)
    #print(contagem)
    if contagem >= limite:
        perguntas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1

respostas_palavras_int = {}
numero_palavra = 0
for palavra, contagem in palavras_contagem.items():
    if contagem >= limite:
        respostas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1
    
# Adição de tokens no dicionário
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    perguntas_palavras_int[token] = len(perguntas_palavras_int) + 1
for token in tokens:
    respostas_palavras_int[token] = len(respostas_palavras_int) + 1

# Criação do dicionário inverso com o dicionário de respostas
respostas_int_palavras = {p_i: p for p, p_i in respostas_palavras_int.items()}

# Adição do token final de string <EOS> para o final de cada resposta
for i in range(len(respostas_limpas)):
    respostas_limpas[i] += ' <EOS>'
    
# Tradução de todas as perguntas e respostas para inteiros
# Substituição das palavras menos frequentes para <OUT>
perguntas_para_int = []
for pergunta in perguntas_limpas:
    ints = []
    for palavra in pergunta.split():
        if palavra not in perguntas_palavras_int:
            ints.append(perguntas_palavras_int['<OUT>'])
        else:
            ints.append(perguntas_palavras_int[palavra])
    perguntas_para_int.append(ints)
        
respostas_para_int = []
for resposta in respostas_limpas:
    ints = []
    for palavra in resposta.split():
        if palavra not in respostas_palavras_int:
            ints.append(respostas_palavras_int['<OUT>'])
        else:
            ints.append(respostas_palavras_int[palavra])
    respostas_para_int.append(ints)
    
# Ordenação das perguntas e respostas pelo tamanho das perguntas
perguntas_limpas_ordenadas = []
respostas_limpas_ordenadas = []
for tamanho in range(1, 25 + 1):
    #print(tamanho)
    for i in enumerate(perguntas_para_int):
        #print(i[1])
        if len(i[1]) == tamanho:
            perguntas_limpas_ordenadas.append(perguntas_para_int[i[0]])
            respostas_limpas_ordenadas.append(respostas_para_int[i[0]])
            
# --- Parte 2 - Construção do modelo Seq2Seq ---            
            
# Criação de placeholders para as entradas e saídas
# [64, 25]
# Olá <PAD> <PAD> <PAD> <PAD>
# Olá tudo bem <PAD> <PAD>
# Olá tudo bem e você
# [3, 5]
def entradas_modelo():
    entradas = tf.placeholder(tf.int32, [None, None], name = 'entradas')
    saidas = tf.placeholder(tf.int32, [None, None], name = 'saidas')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return entradas, saidas, lr, keep_prob

# Pré-processamento das saídas (alvos)
# [batch_size, 1] = [64, 1]
# 0 - SOS (8825)
# 1 - SOS (8825)
def preprocessamento_saidas(saidas, palavra_para_int, batch_size):
    esquerda = tf.fill([batch_size, 1], palavra_para_int['<SOS>'])
    direita = tf.strided_slice(saidas, [0,0], [batch_size, -1], strides = [1,1])
    saidas_preprocessadas = tf.concat([esquerda, direita], 1)
    return saidas_preprocessadas

# Criação da RNN do codificador
#tf.VERSION    
def rnn_codificador(rnn_entradas, rnn_tamanho, numero_camadas, keep_prob, tamanho_sequencia):
    lstm = tf.contrib.rnn.LSTMCell(rnn_tamanho)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_celula = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * numero_camadas)
    _, encoder_estado = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_celula,
                                                     cell_bw = encoder_celula,
                                                     sequence_length = tamanho_sequencia,
                                                     inputs = rnn_entradas,
                                                     dtype = tf.float32)
    return encoder_estado

# Decodificação da base de treinamento
def decodifica_base_treinamento(encoder_estado, decodificador_celula, 
                                decodificador_embedded_entrada, tamanho_sequencia,
                                decodificador_escopo, funcao_saida,
                                keep_prob, batch_size):
    estados_atencao = tf.zeros([batch_size, 1, decodificador_celula.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(estados_atencao,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units = decodificador_celula.output_size)
    funcao_decodificador_treinamento = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_estado[0],
                                                                                     attention_keys, 
                                                                                     attention_values, 
                                                                                     attention_score_function, 
                                                                                     attention_construct_function,
                                                                                     name = 'attn_dec_train')
    decodificador_saida, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decodificador_celula,
                                                                       funcao_decodificador_treinamento,
                                                                       decodificador_embedded_entrada,
                                                                       tamanho_sequencia,
                                                                       scope = decodificador_escopo)
    decodificador_saida_dropout = tf.nn.dropout(decodificador_saida, keep_prob)
    return funcao_saida(decodificador_saida_dropout)  
    
# Decodificação da base de teste/validação
def decodifica_base_teste(encoder_estado, decodificador_celula, 
                          decodificador_embedding_matrix,sos_id, eos_id, tamanho_maximo,
                          numero_palavras, decodificador_escopo, funcao_saida,
                          keep_prob, batch_size):                          
    estados_atencao = tf.zeros([batch_size, 1, decodificador_celula.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(estados_atencao,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units = decodificador_celula.output_size)
    funcao_decodificador_teste = tf.contrib.seq2seq.attention_decoder_fn_inference(funcao_saida,
                                                                                   encoder_estado[0],
                                                                                   attention_keys, 
                                                                                  attention_values, 
                                                                                   attention_score_function, 
                                                                                   attention_construct_function,
                                                                                   decodificador_embedding_matrix,
                                                                                   sos_id,
                                                                                   eos_id,
                                                                                   tamanho_maximo,
                                                                                   numero_palavras,
                                                                                   name = 'attn_dec_inf')
    previsoes_teste, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decodificador_celula,
                                                                   funcao_decodificador_teste,
                                                                   scope = decodificador_escopo)
    return previsoes_teste  

# Criação da RNN do decodificador
def rnn_decodificador(decodificador_embedded_entrada, decodificador_embeddings_matrix,
                      codificador_estado, numero_palavras, tamanho_sequencia, rnn_tamanho,
                      numero_camadas, palavra_para_int, keep_prob, batch_size):
    with tf.variable_scope("decodificador") as decodificador_escopo:
        lstm = tf.contrib.rnn.LSTMCell(rnn_tamanho)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decodificador_celula = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * numero_camadas)
        pesos = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        funcao_saida = lambda x: tf.contrib.layers.fully_connected(x, numero_palavras,
                                                                   None,
                                                                   scope = decodificador_escopo,
                                                                   weights_initializer = pesos,
                                                                   biases_initializer = biases)
        previsoes_treinamento = decodifica_base_treinamento(codificador_estado,
                                                            decodificador_celula,
                                                            decodificador_embedded_entrada,
                                                            tamanho_sequencia,
                                                            decodificador_escopo,
                                                            funcao_saida,
                                                            keep_prob,
                                                            batch_size)
        decodificador_escopo.reuse_variables()
        previsoes_teste = decodifica_base_teste(codificador_estado,
                                                decodificador_celula,
                                                decodificador_embeddings_matrix,
                                                palavra_para_int['<SOS>'],
                                                palavra_para_int['<EOS>'],
                                                tamanho_sequencia - 1,
                                                numero_palavras,
                                                decodificador_escopo,
                                                funcao_saida,
                                                keep_prob,
                                                batch_size)
        return previsoes_treinamento, previsoes_teste
    
# Criação do modelo Seq2Seq
def modelo_seq2seq(entradas, saidas, keep_prob, batch_size, tamanho_sequencia,
                   numero_palavras_respostas, numero_palavras_perguntas,
                   tamanho_codificador_embeddings, tamanho_decodificador_embeddings,
                   rnn_tamanho, numero_camadas, perguntas_palavras_int):
    codificador_embedded_entrada = tf.contrib.layers.embed_sequence(entradas,
                                                                    numero_palavras_respostas + 1,
                                                                    tamanho_codificador_embeddings,
                                                                    initializer = tf.random_uniform_initializer(0,1))
    codificador_estado = rnn_codificador(codificador_embedded_entrada,
                                         rnn_tamanho, numero_camadas,
                                         keep_prob, tamanho_sequencia)
    saidas_preprocessadas = preprocessamento_saidas(saidas, perguntas_palavras_int, batch_size)
    decodificador_embeddings_matrix = tf.Variable(tf.random_uniform([numero_palavras_perguntas + 1,
                                                                     tamanho_decodificador_embeddings], 0, 1))
    decodificador_embedded_entradas = tf.nn.embedding_lookup(decodificador_embeddings_matrix,
                                                             saidas_preprocessadas)
    previsoes_treinamento, previsoes_teste = rnn_decodificador(decodificador_embedded_entradas,
                                                               decodificador_embeddings_matrix,
                                                               codificador_estado,
                                                               numero_palavras_perguntas,
                                                               tamanho_sequencia,
                                                               rnn_tamanho,
                                                               numero_camadas,
                                                               perguntas_palavras_int,
                                                               keep_prob,
                                                               batch_size)
    return previsoes_treinamento, previsoes_teste
    
# --- Parte 3 - Treinamento do modelo Seq2Seq ---   

# Configuração dos hiperparâmetros
epocas = 100
batch_size = 32
rnn_tamanho = 1024
numero_camadas = 3
tamanho_codificador_embeddings = 1024
tamanho_decodificador_embeddings = 1024
learning_rate = 0.001
learning_rate_decaimento = 0.9
min_learning_rate = 0.0001
probabilidade_dropout = 0.5

# Definição da seção
tf.reset_default_graph()
session = tf.InteractiveSession()

# Carregamento do modelo
entradas, saidas, lr, keep_prob = entradas_modelo()

# Configuração do tamanho da sequência
tamanho_sequencia = tf.placeholder_with_default(25, None, name = 'tamanho_sequencia')
     
# Obtenção das dimensões dos tensores de entrada
dimensao_entrada = tf.shape(entradas) 

# Obtenção das previsões de treinamento e teste
previsoes_treinamento, previsoes_teste = modelo_seq2seq(tf.reverse(entradas, [-1]),
                                                        saidas,
                                                        keep_prob,
                                                        batch_size,
                                                        tamanho_sequencia,
                                                        len(respostas_palavras_int),
                                                        len(perguntas_palavras_int),
                                                        tamanho_codificador_embeddings,
                                                        tamanho_decodificador_embeddings,
                                                        rnn_tamanho,
                                                        numero_camadas,
                                                        perguntas_palavras_int)

# Loss function (erro), otimizador e gradient clipping
with tf.name_scope("otimizacao"):
    erro = tf.contrib.seq2seq.sequence_loss(previsoes_treinamento, saidas,
                                            tf.ones([dimensao_entrada[0], tamanho_sequencia]))
    otimizador = tf.train.AdamOptimizer(learning_rate)
    gradients = otimizador.compute_gradients(erro)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    otimizador_clipping = otimizador.apply_gradients(clipped_gradients)
    
# Padding
# Olá <PAD> <PAD> <PAD> <PAD>
# Olá tudo bem <PAD> <PAD>
# Olá tudo bem e você
def aplica_padding(batch_textos, palavra_para_int):
    tamanho_maximo = max([len(texto) for texto in batch_textos])
    return [texto + [palavra_para_int['<PAD>']] * (tamanho_maximo - len(texto)) for texto in batch_textos]

# Divisão dos dados em batches de perguntas e respostas
def divide_batches(perguntas, respostas, batch_size):
    for indice_batch in range(0, len(perguntas) // batch_size):
        indice_inicio = indice_batch * batch_size
        perguntas_no_batch = perguntas[indice_inicio:indice_inicio + batch_size]
        respostas_no_batch = respostas[indice_inicio:indice_inicio + batch_size]
        perguntas_no_batch_padded = np.array(aplica_padding(perguntas_no_batch, perguntas_palavras_int))
        respostas_no_batch_padded = np.array(aplica_padding(respostas_no_batch, respostas_palavras_int))
        yield perguntas_no_batch_padded, respostas_no_batch_padded
        
# Divisão das perguntas e respostas em base de treinamento e teste/validação
indice_base_validacao = int(len(perguntas_limpas_ordenadas) * 0.15)
perguntas_treinamento = perguntas_limpas_ordenadas[indice_base_validacao:]
respostas_treinamento = respostas_limpas_ordenadas[indice_base_validacao:]
perguntas_validacao = perguntas_limpas_ordenadas[:indice_base_validacao]
respostas_validacao = respostas_limpas_ordenadas[:indice_base_validacao]

# Treinamento
batch_indice_checagem_treinamento = 20
batch_indice_checagem_validacao = (len(perguntas_treinamento) // batch_size // 2) - 1
erro_total_treinamento = 0
lista_validacao_erro = []
early_stopping_checagem = 0
early_stopping_parada = 1000
checkpoint = "chatbot_pesos.ckpt" # ./
session.run(tf.global_variables_initializer())
for epoca in range(1, epocas + 1):
    for indice_batch, (perguntas_no_batch_padded, respostas_no_batch_padded) in enumerate(divide_batches(perguntas_treinamento, respostas_treinamento, batch_size)):
        tempo_inicio = time.time()
        _, erro_treinamento_batch = session.run([otimizador_clipping, erro], feed_dict = {entradas: perguntas_no_batch_padded,
                                                    saidas: respostas_no_batch_padded, 
                                                    lr: learning_rate,
                                                    tamanho_sequencia: respostas_no_batch_padded.shape[1],
                                                    keep_prob: probabilidade_dropout})
        erro_total_treinamento += erro_treinamento_batch
        tempo_final = time.time()
        tempo_batch = tempo_final - tempo_inicio
        if indice_batch % batch_indice_checagem_treinamento == 0:
            print('Época: {:>3}/{}, Batch: {:>4}/{}, Erro treinamento: {:>6.3f}, Tempo treinamento em 100 batches: {:d} segundos'.format(epoca,
                  epocas,
                  indice_batch,
                  len(perguntas_treinamento) // batch_size,
                  erro_total_treinamento / batch_indice_checagem_treinamento,
                  int(tempo_batch * batch_indice_checagem_treinamento)))
        
            erro_total_treinamento = 0
        
        if indice_batch % batch_indice_checagem_validacao == 0 and indice_batch > 0:
            erro_total_validacao = 0
            tempo_inicio = time.time()
            for indice_batch_validacao, (perguntas_no_batch_padded, respostas_no_batch_padded) in enumerate(divide_batches(perguntas_validacao, respostas_validacao, batch_size)):
                erro_batch_validacao = session.run(erro, feed_dict = {entradas: perguntas_no_batch_padded,
                                                    saidas: respostas_no_batch_padded, 
                                                    lr: learning_rate,
                                                    tamanho_sequencia: respostas_no_batch_padded.shape[1],
                                                    keep_prob: 1})
                erro_total_validacao += erro_batch_validacao
            tempo_final = time.time()
            tempo_batch = tempo_final - tempo_inicio
            media_erro_validacao = erro_total_validacao / (len(perguntas_validacao) / batch_size)
            print('Erro validação: {:>6.3f}, Tempo validação batch: {:d} segundos'.format(media_erro_validacao, int(tempo_batch)))
            learning_rate *= learning_rate_decaimento
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            lista_validacao_erro.append(media_erro_validacao)
            if media_erro_validacao < min(lista_validacao_erro):
                print('Houve melhoria')
                early_stopping_checagem = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
               print('Eu não consigo falar melhor! Preciso praticar mais')
               early_stopping_checagem += 1
               if early_stopping_checagem == early_stopping_parada:
                   break
               
    if early_stopping_checagem == early_stopping_parada:
        print('Isso é o melhor que eu posso fazer')
        break
print('Final')
            
# --- Parte 4 - Testes com o modelo Seq2Seq ---

# Carregamento dos pesos e executando a seção
checkpoint = "./chatbot_pesos.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Conversão de questões de string para inteiros
def converte_string_para_int(pergunta, palavra_para_int):
    pergunta = limpa_texto(pergunta)
    return [palavra_para_int.get(palavra, palavra_para_int['<OUT>']) for palavra in pergunta.split()]

converte_string_para_int("i'm a robot", perguntas_palavras_int)

# Conversa
while(True):
    pergunta = input('Você: ')
    if pergunta == 'Tchau':
        break
    pergunta = converte_string_para_int(pergunta, perguntas_palavras_int)
    pergunta = pergunta + [perguntas_palavras_int['<PAD>']] * (25 - len(pergunta))
    # [64, 25]
    batch_falso = np.zeros((batch_size, 25))
    batch_falso[0] = pergunta
    resposta_prevista = session.run(previsoes_teste, feed_dict = {entradas: batch_falso,
                                                                   keep_prob: 1})[0]
    respostas = ''
    for i in np.argmax(resposta_prevista, 1):
        if respostas_int_palavras[i] == 'i':
            token = 'I'
        elif respostas_int_palavras[i] == '<EOS>':
            token = '.'
        elif respostas_int_palavras[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + respostas_int_palavras[i]
        resposta += token
        if token == '.':
            break
    print('Chatbot: ' + resposta)