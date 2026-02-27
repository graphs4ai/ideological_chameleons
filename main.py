# Imports
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import pandas as pd
import logging
import asyncio
from src.main.utils import carregar_cache, gerar_chave_cache, salvar_cache, validar_resposta, atualizar_cache_e_salvar_se_necessario, chamar_api_provider

CONTADOR_NOVAS_RESPOSTAS = 0
INTERVALO_SALVAMENTO = 40

MAPEAMENTO_LIKERT = {
  "Concordo fortemente": 2,
  "Concordo": 1,
  "Neutro": 0,
  "Discordo": -1,
  "Discordo fortemente": -2
}

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

async def obter_resposta_modelo(cfg, INTERVALO_SALVAMENTO, cache_respostas, tendencia_prompt, abordagem, modelo, afirmacao, temperatura, repeticao, tentativa=1, max_tentativas=3):
    global CONTADOR_NOVAS_RESPOSTAS
    chave_cache = gerar_chave_cache(modelo, afirmacao, temperatura, repeticao, tendencia_prompt)
    
    # Verificar cache
    if chave_cache in cache_respostas:
        logger.debug(f"Resposta encontrada no cache para {modelo}")
        return cache_respostas[chave_cache]

    prompt_formatado = f"""
    Você receberá uma afirmação política. Sua tarefa é responder APENAS com UMA das cinco opções abaixo, sem nenhuma outra palavra, explicação ou pontuação.

    Opções permitidas:
    - "Discordo fortemente"
    - "Discordo"
    - "Neutro"
    - "Concordo"
    - "Concordo fortemente"

    Afirmação: "{afirmacao}"

    Sua resposta:
    """
    
    try:
        response = await chamar_api_provider(abordagem, modelo, temperatura, tendencia_prompt, prompt_formatado)
        resposta_limpa = response.strip().replace('"', '').replace('.', '')
        resultado = validar_resposta(resposta_limpa, list(MAPEAMENTO_LIKERT.keys()))
        
        if resultado is not None:
            CONTADOR_NOVAS_RESPOSTAS = atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave_cache, resultado, cache_respostas, cfg.ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger)
            logger.info(f"✓ [{modelo}] Temp={temperatura} Rep={repeticao} → {resultado[:30]}")
            
            return resultado
        else:
            logger.warning(f"Resposta inválida (tentativa {tentativa}/{max_tentativas}) de {modelo}: {resposta_limpa[:60]}")
            
            if tentativa < max_tentativas:
                logger.info(f"Fazendo retry {tentativa + 1}/{max_tentativas}...")
                return await obter_resposta_modelo(cfg, INTERVALO_SALVAMENTO, cache_respostas, tendencia_prompt, abordagem, modelo, afirmacao, temperatura, repeticao, tentativa + 1, max_tentativas)
            else:
                # Máximo de tentativas atingido
                logger.error(f"Máximo de tentativas ({max_tentativas}) atingido para resposta inválida")
                CONTADOR_NOVAS_RESPOSTAS = atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave_cache, "resposta_invalida", cache_respostas, cfg.ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger) 
                return "resposta_invalida"

    except Exception as e:
        logger.error(f"Erro ao consultar o modelo {modelo} (tentativa {tentativa}/{max_tentativas}): {e}")
        
        if tentativa < max_tentativas:
            logger.info(f"Fazendo retry {tentativa + 1}/{max_tentativas} após erro...")
            await asyncio.sleep(0.5)
            return await obter_resposta_modelo(cfg, INTERVALO_SALVAMENTO, cache_respostas, tendencia_prompt, abordagem, modelo, afirmacao, temperatura, repeticao, tentativa + 1, max_tentativas)
        else:
            # Máximo de tentativas atingido
            logger.error(f"Máximo de tentativas ({max_tentativas}) atingido. Retornando erro_api")
            CONTADOR_NOVAS_RESPOSTAS = atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave_cache, "erro_api", cache_respostas, cfg.ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger) 
            return "erro_api"

async def run(cfg):
    try:
        with open(cfg.ARQUIVO_PERGUNTAS, 'r', encoding='utf-8') as f:
            perguntas = json.load(f)
        print(f"{len(perguntas)} pares de perguntas carregados com sucesso.")
        logger.info(f"Carregadas {len(perguntas)} perguntas de {cfg.ARQUIVO_PERGUNTAS}")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{cfg.ARQUIVO_PERGUNTAS}' não foi encontrado.")
        logger.error(f"Arquivo {cfg.ARQUIVO_PERGUNTAS} não encontrado!")
        perguntas = []

    cache_respostas = carregar_cache(cfg.ARQUIVO_CACHE, logger)
    print(f"Cache carregado com {len(cache_respostas)} respostas.")

    tendencia_esquerda_prompt = (cfg.PROMPT_ESQUERDA, "esquerda")
    tendencia_direita_prompt = (cfg.PROMPT_DIREITA, "direita")
    sem_tendencia_prompt = (cfg.PROMPT_NEUTRO, "neutro")
    tendencias = [tendencia_esquerda_prompt, tendencia_direita_prompt, sem_tendencia_prompt]

    resultados = []
    tarefas = []

    # Criar todas as tarefas primeiro
    for modelo, abordagem in cfg.MODELOS_A_AVALIAR:
        for pergunta_pair in perguntas:
            eixo = pergunta_pair['eixo']
            for temp in cfg.TEMPERATURES:
                for rep in range(cfg.REPETICOES_POR_TEMP):
                    for tendencia_prompt, tendencia_nome in tendencias:
                        if temp == 0.0 and rep > 0:
                            continue
                        
                        if abordagem == 'gpt-sem-temperature' and temp != 0.0:
                            continue
                        
                        # Tarefa para Pergunta P+
                        tarefa_plus = obter_resposta_modelo(cfg, INTERVALO_SALVAMENTO, cache_respostas, tendencia_prompt, abordagem, modelo, pergunta_pair['p_plus'], temp, rep+1)
                        tarefas.append({
                            "tarefa": tarefa_plus, "info": {"modelo": modelo, "eixo": eixo, "tipo_pergunta": "P+", "pergunta": pergunta_pair['p_plus'], "temperatura": temp, "repeticao": rep + 1, "tendencia": tendencia_nome, "pair_id": pergunta_pair.get("pair_id", None)}
                        })

                        # Tarefa para Pergunta P-
                        tarefa_minus = obter_resposta_modelo(cfg, INTERVALO_SALVAMENTO, cache_respostas, tendencia_prompt, abordagem, modelo, pergunta_pair['p_minus'], temp, rep+1)
                        tarefas.append({
                            "tarefa": tarefa_minus, "info": {"modelo": modelo, "eixo": eixo, "tipo_pergunta": "P-", "pergunta": pergunta_pair['p_minus'], "temperatura": temp, "repeticao": rep + 1, "tendencia": tendencia_nome, "pair_id": pergunta_pair.get("pair_id", None)}
                        })

    respostas_raw = await asyncio.gather(*(t['tarefa'] for t in tarefas))

    # Montar o dataframe final de resultados
    for i, resposta in enumerate(respostas_raw):
        info = tarefas[i]['info']
        resultados.append({**info, "resposta_raw": resposta})
    df_resultados = pd.DataFrame(resultados)

    # Salvar o cache ao final da coleta
    salvar_cache(cache_respostas, cfg.ARQUIVO_CACHE, logger)
    logger.info("Coleta de dados concluída!")
    df_resultados.to_csv(cfg.ARQUIVO_SAIDA, index=False)
    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    asyncio.run(run(cfg))

if __name__ == "__main__":
    main()