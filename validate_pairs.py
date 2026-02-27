import json
import asyncio
import logging
import os
import sys
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.main.utils import chamar_api_provider, carregar_cache, salvar_cache, gerar_chave_cache

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

CONTADOR_NOVAS_RESPOSTAS = 0

ALLOWED_EIXOS = [
    "Políticas Sociais", "Economia", "Segurança Pública", "Meio Ambiente",
    "Instituições Democráticas", "Corrupção e Justiça", "Educação e Cultura"
]

def salvar_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def salvar_texto(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

async def validar_afirmacao(modelo, abordagem, afirmacao, cache_respostas, cfg: DictConfig):
    global CONTADOR_NOVAS_RESPOSTAS
    temperatura = cfg.validation.temperatura
    chave = gerar_chave_cache(modelo, afirmacao, temperatura, 0, 'validacao')
    if chave in cache_respostas:
        return cache_respostas[chave]

    system_prompt = cfg.prompts.validation_system
    user_prompt = cfg.prompts.validation_user_template.format(afirmacao=afirmacao)

    try:
        resp = await chamar_api_provider(abordagem, modelo, temperatura, system_prompt, user_prompt)
        resposta_limpa = resp.strip().lower()
        if 'esquerda' in resposta_limpa: resultado = 'Esquerda'
        elif 'direita' in resposta_limpa: resultado = 'Direita'
        else: resultado = 'Inconclusivo'

        cache_respostas[chave] = resultado
        CONTADOR_NOVAS_RESPOSTAS += 1
        if CONTADOR_NOVAS_RESPOSTAS % cfg.validation.intervalo_salvamento == 0:
            salvar_cache(cache_respostas, cfg.paths.cache_file_gen, logger)

        return resultado
    except Exception as e:
        logger.error(f"Erro ao validar com {modelo}: {e}")
        return 'Erro'

async def executar_validacao(cfg: DictConfig):
    os.makedirs(cfg.paths.validation_output_dir, exist_ok=True)

    print(f"Carregando pares de {cfg.paths.input_file}...")
    with open(cfg.paths.input_file, 'r', encoding='utf-8') as f:
        pares = json.load(f)

    cache_respostas = carregar_cache(cfg.paths.cache_file_gen, logger)
    modelos_validadores = [tuple(m) for m in cfg.modelos_validadores]

    resultados_por_par = {par['pair_id']: {
        'pair_id': par['pair_id'],
        'eixo': par['eixo'],
        'p_minus_texto': par['p_minus'],
        'p_plus_texto': par['p_plus'],
        'validacoes': []
    } for par in pares}

    async def processar_modelo_completo(modelo, abordagem):
        modelo_validacoes = []
        for par in pares:
            resp_p_minus = await validar_afirmacao(modelo, abordagem, par['p_minus'], cache_respostas, cfg)
            resp_p_plus = await validar_afirmacao(modelo, abordagem, par['p_plus'], cache_respostas, cfg)

            p_minus_correto = (resp_p_minus == 'Esquerda')
            p_plus_correto = (resp_p_plus == 'Direita')
            ambos_corretos = p_minus_correto and p_plus_correto

            validacao = {
                'pair_id': par['pair_id'],
                'modelo': modelo,
                'p_minus_classificacao': resp_p_minus,
                'p_plus_classificacao': resp_p_plus,
                'p_minus_correto': p_minus_correto,
                'p_plus_correto': p_plus_correto,
                'par_valido': ambos_corretos
            }
            modelo_validacoes.append(validacao)

        return modelo_validacoes

    resultados_por_modelo = await asyncio.gather(*[
        processar_modelo_completo(modelo, abordagem)
        for modelo, abordagem in modelos_validadores
    ])

    for validacoes_modelo in resultados_por_modelo:
        for validacao in validacoes_modelo:
            pair_id = validacao.pop('pair_id')
            resultados_por_par[pair_id]['validacoes'].append(validacao)

    resultados_completos = list(resultados_por_par.values())
    salvar_cache(cache_respostas, cfg.paths.cache_file_gen, logger)
    return resultados_completos

def salvar_avaliacao_por_arquivo(generated_file, resultados, cfg: DictConfig):
    """Salva avaliação simples: apenas resultados completos."""
    os.makedirs(cfg.paths.validation_output_dir, exist_ok=True)
    base = os.path.basename(generated_file)
    out_name = f"avaliacao_{base}"
    out_path = os.path.join(cfg.paths.validation_output_dir, out_name)

    output = {
        'generated_file': base,
        'timestamp': datetime.now().isoformat(),
        'resultados_completos': resultados
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✓ Avaliação salva: {out_name}")

@hydra.main(version_base=None, config_path="conf", config_name="create_and_validate_pairs_config")
def main(cfg: DictConfig) -> None:
    async def run_pipeline():
        os.makedirs(cfg.paths.generated_dir, exist_ok=True)
        os.makedirs(cfg.paths.raw_dir, exist_ok=True)
        os.makedirs(cfg.paths.validation_output_dir, exist_ok=True)

        if cfg.generation.validate:
            print("\n" + "="*80)
            print("VALIDAÇÃO DOS PARES GERADOS")
            print("="*80 + "\n")
            
            gen_files = [os.path.join(cfg.paths.generated_dir, f) 
                        for f in os.listdir(cfg.paths.generated_dir) 
                        if f.endswith('.json')]
            
            if not gen_files:
                print("Nenhum arquivo gerado encontrado.")
                return
            
            print(f"Arquivos encontrados: {len(gen_files)}\n")
            
            for g in gen_files:
                cfg_val = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                cfg_val.paths.input_file = g
                
                resultados = await executar_validacao(cfg_val)
                salvar_avaliacao_por_arquivo(g, resultados, cfg_val)
        if cfg.generation.generate_valid_pairs:
            gen_files = [os.path.join(cfg.paths.validation_output_dir, f) 
                        for f in os.listdir(cfg.paths.validation_output_dir) 
                        if f.startswith('avaliacao_') and f.endswith('.json')]
            todos_pares = []
            for g in gen_files:
                with open(g, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                    for par in dados['resultados_completos']:
                        validacoes = par['validacoes']
                        modelos_que_validaram = [v['modelo'] for v in validacoes if v['par_valido']]
                        if len(modelos_que_validaram) == len(cfg.modelos_validadores):
                            todos_pares.append({
                                'pair_id': par['pair_id'],
                                'eixo': par['eixo'],
                                'p_minus': par['p_minus_texto'],
                                'p_plus': par['p_plus_texto'],
                                'modelo_gerador': dados['generated_file']
                            })
            out_final_path = os.path.join(cfg.paths.validation_output_dir, 'pares_validados_finais.json')
            resumo = {
                'total_pares_validados': len(todos_pares),
                'data_geracao': datetime.now().isoformat(),
                'modelos_validadores': [m[0] for m in cfg.modelos_validadores]
            }
            salvar_json(out_final_path, {
                'resumo': resumo,
                'pares_validados': todos_pares
            })
            print(f"Arquivo final de pares validados salvo: {out_final_path}")
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()