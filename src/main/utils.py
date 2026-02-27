import hashlib
import os
import pickle
from openai import OpenAI, AsyncOpenAI
from xai_sdk import Client
from xai_sdk.chat import user, system
from dotenv import load_dotenv
from google import genai
import ollama
import asyncio

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
MARITACA_API_KEY = os.getenv("MARITACA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

# Inicializar clientes de API
client_deepinfra = AsyncOpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
client_grok = Client(api_key=GROK_API_KEY)
client_genai = genai.Client(api_key=GEMINI_API_KEY)
client_ollama = ollama.Client(host=OLLAMA_HOST)
client_openai = OpenAI(api_key=OPEN_AI_API_KEY)
cliente_maritaca = OpenAI(api_key=MARITACA_API_KEY, base_url="https://chat.maritaca.ai/api")
    

gemini_semaphore = asyncio.Semaphore(10)
deepinfra_semaphore = asyncio.Semaphore(10)

def gerar_chave_cache(modelo, afirmacao, temperatura, repeticao, tendencia_prompt):
    """Gera uma chave única incluindo a repetição."""
    chave = f"{modelo}|{afirmacao}|{temperatura}|{repeticao}|{tendencia_prompt}"
    return hashlib.md5(chave.encode()).hexdigest()

def carregar_cache(ARQUIVO_CACHE, logger):
    """Carrega o cache de respostas anteriores."""
    if os.path.exists(ARQUIVO_CACHE):
        try:
            with open(ARQUIVO_CACHE, 'rb') as f:
                cache = pickle.load(f)
                logger.info(f"Cache carregado com {len(cache)} entradas")
                return cache
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")
            return {}
    return {}

def salvar_cache(cache, ARQUIVO_CACHE, logger):
    """Salva o cache de respostas."""
    try:
        with open(ARQUIVO_CACHE, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Cache salvo com {len(cache)} entradas")
    except Exception as e:
        logger.error(f"Erro ao salvar cache: {e}")

def validar_resposta(resposta_limpa, OPCOES_VALIDAS):
    """
    Valida se a resposta está entre as opções permitidas.
    Retorna a resposta válida ou None se inválida.
    """
    # Verificação exata
    if resposta_limpa in OPCOES_VALIDAS:
        return resposta_limpa
    
    # Tentar encontrar uma opção válida na resposta (fallback)
    for opcao in OPCOES_VALIDAS:
        if opcao in resposta_limpa:
            return opcao
    
    return None

def atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave, valor, cache_respostas, ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger):
    cache_respostas[chave] = valor
    CONTADOR_NOVAS_RESPOSTAS += 1
    if CONTADOR_NOVAS_RESPOSTAS % INTERVALO_SALVAMENTO == 0:
        logger.info(f"Salvamento incremental ({CONTADOR_NOVAS_RESPOSTAS} novas respostas)...")
        try:
            salvar_cache(cache_respostas, ARQUIVO_CACHE, logger) 
        except Exception as e:
            logger.error(f"Erro no salvamento incremental: {e}")
    return CONTADOR_NOVAS_RESPOSTAS


async def chamar_api_provider(abordagem, modelo, temperatura, system_prompt, user_prompt):
    response_content = ""
    if abordagem == 'ollama':
        response = client_ollama.chat(
            model=modelo,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            options={'temperature': temperatura}
        )
        response_content = response['message']['content']

    elif abordagem == 'gemini':
        async with gemini_semaphore:
            gemini_model = client_genai.GenerativeModel(model_name=modelo)
            gemini_resposta = await gemini_model.generate_content_async(
                [{"role": "model", "parts": system_prompt}, {"role": "user", "parts": user_prompt}],
                generation_config=client_genai.types.GenerationConfig(temperature=temperatura)
            )
            response_content = gemini_resposta.text

    elif abordagem in ['gpt', 'gpt-sem-temperature']:
        kwargs = {
            "model": modelo,
            "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        }
        if abordagem == 'gpt':
             kwargs["temperature"] = temperatura
        
        gpt_resposta = client_openai.responses.create(**kwargs)
        response_content = gpt_resposta.output_text
    elif abordagem == 'deepinfra':
        async with deepinfra_semaphore:
            deepinfra_resposta = await client_deepinfra.chat.completions.create(
                model=modelo,
                temperature=temperatura,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_content = deepinfra_resposta.choices[0].message.content
    elif abordagem == 'maritaca':
        maritaca_resposta = cliente_maritaca.chat.completions.create(
            model=modelo,
            temperature=temperatura,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        response_content = maritaca_resposta.choices[0].message.content
    elif abordagem == 'grok':
        chat = client_grok.chat.create(model=modelo, temperature=temperatura)
        chat.append(system(system_prompt))
        chat.append(user(user_prompt))
        grok_resposta = chat.sample()
        response_content = grok_resposta.content

    return response_content