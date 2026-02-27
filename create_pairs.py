import json
import asyncio
import logging
import os
import re
import sys

import hydra
from omegaconf import DictConfig

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

def extract_first_json_array(text: str):
    start = text.find('[')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                try:
                    candidate = text[start:i+1]
                    return json.loads(candidate)
                except Exception:
                    return None
    return None

def validate_generated_structure(pairs):
    if not isinstance(pairs, list):
        return False, "Not a list"
    if len(pairs) != 49:
        return False, f"Expected 49 objects, got {len(pairs)}"
    ids = set()
    eixo_counts = {e: 0 for e in ALLOWED_EIXOS}
    for obj in pairs:
        if not all(k in obj for k in ("pair_id", "eixo", "p_minus", "p_plus")):
            return False, "Missing keys in an object"
        pid = obj['pair_id']
        if not isinstance(pid, int) or not (0 <= pid <= 48):
            return False, f"Invalid pair_id: {pid}"
        ids.add(pid)
        eixo = obj['eixo']
        if eixo not in ALLOWED_EIXOS:
            return False, f"Invalid eixo: {eixo}"
        eixo_counts[eixo] += 1
        if not isinstance(obj['p_minus'], str) or not isinstance(obj['p_plus'], str):
            return False, "p_minus/p_plus must be strings"
    if len(ids) != 49:
        return False, "pair_id values must be unique and cover 0..48"
    for e, count in eixo_counts.items():
        if count != 7:
            return False, f"Eixo {e} has {count} pairs (expected 7)"
    return True, "OK"

async def gerar_pares_com_modelo(modelo, abordagem, cfg):
    temperatura = float(cfg.generation.temperatura)
    chave = gerar_chave_cache(modelo, 'geracao_pares', temperatura, 0, 'geracao')

    cache = carregar_cache(cfg.paths.cache_file_gen, logger)
    if chave in cache:
        logger.info(f"Cache encontrado para {modelo}, usando resposta em cache")
        parsed = cache[chave]
        return parsed, None

    system_prompt = cfg.prompts.generation_system
    user_template = cfg.prompts.generation_user_template

    try:
        resposta = await chamar_api_provider(abordagem, modelo, temperatura, system_prompt, user_template)
    except Exception as e:
        logger.error(f"Erro ao chamar o modelo {modelo}: {e}")
        return None, None

    parsed = extract_first_json_array(resposta)
    cache[chave] = parsed
    salvar_cache(cache, cfg.paths.cache_file_gen, logger)
    return parsed, resposta

def salvar_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def salvar_texto(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

@hydra.main(version_base=None, config_path="conf", config_name="create_and_validate_pairs_config")
def main(cfg: DictConfig) -> None:
    async def run_pipeline():
        os.makedirs(cfg.paths.generated_dir, exist_ok=True)
        os.makedirs(cfg.paths.raw_dir, exist_ok=True)

        modelos_geradores = [tuple(m) for m in cfg.generation.modelos_geradores]

        if cfg.generation.generate:
            print("\n" + "="*80)
            print("GERAÇÃO DE PARES DE AFIRMAÇÕES POLÍTICAS")
            print("="*80 + "\n")

            for modelo, abordagem in modelos_geradores:
                print(f"Gerando pares com o modelo {modelo}...")
                parsed, raw = await gerar_pares_com_modelo(modelo, abordagem, cfg)
                print(f"Validação da estrutura dos pares gerados por {modelo}...")
                print(raw[:100] + "...\n" if raw else "No raw response.\n")
                safe = re.sub(r"[^0-9a-zA-Z]+", "_", modelo)
                out_json = os.path.join(cfg.paths.generated_dir, f"afirmacoes_{safe}.json")
                raw_path = os.path.join(cfg.paths.raw_dir, f"raw_{safe}.txt")

                if isinstance(parsed, list):
                    valid, reason = validate_generated_structure(parsed)
                    if valid:
                        salvar_json(out_json, parsed)
                        print(f"✓ {out_json} salvo")
                    else:
                        print(f"⚠ Validação da estrutura falhou para {modelo}: {reason}")
                        if raw:
                            salvar_texto(raw_path, raw)
                        continue
                else:
                    print(f"✖ Não foi possível parsear JSON de {modelo}. Verifique o arquivo raw: {raw_path}")
                    if raw:
                        salvar_texto(raw_path, raw)
        else :
            print("\nGeração de pares está desabilitada no config.\n")
        print("\n✅ Pipeline concluído!\n")

    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()