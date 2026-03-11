import pandas as pd
from omegaconf import DictConfig

def carregar_e_processar_dados(cfg: DictConfig) -> pd.DataFrame:
    df_resultados = pd.read_csv(cfg.paths.input_file)
    likert_map = dict(cfg.analysis.likert_map)
    
    df_resultados['pontuacao'] = df_resultados['resposta_raw'].map(likert_map)
    df_validos = df_resultados.dropna(subset=['pontuacao']).copy()
    df_validos['pontuacao'] = df_validos['pontuacao'].astype(int)
    return df_validos

def carregar_e_processar_ambos(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both original and negated CSVs for comparison figures."""
    likert_map = dict(cfg.analysis.likert_map)

    df_orig = pd.read_csv("dados/respostas_finais.csv")
    df_orig['pontuacao'] = df_orig['resposta_raw'].map(likert_map)
    df_orig = df_orig.dropna(subset=['pontuacao']).copy()
    df_orig['pontuacao'] = df_orig['pontuacao'].astype(int)

    df_neg = pd.read_csv("dados/respostas_finais_negados.csv")
    df_neg['pontuacao'] = df_neg['resposta_raw'].map(likert_map)
    df_neg = df_neg.dropna(subset=['pontuacao']).copy()
    df_neg['pontuacao'] = df_neg['pontuacao'].astype(int)

    return df_orig, df_neg

def calcular_indice_polarizacao(df_validos: pd.DataFrame):
    # Agrupa médias
    cols_group = ['modelo', 'eixo', 'pair_id', 'tipo_pergunta', 'temperatura', 'tendencia']
    df_medias = df_validos.groupby(cols_group)['pontuacao'].mean().reset_index()

    # Separa P+ e P-
    df_p_plus = df_medias[df_medias['tipo_pergunta'] == 'P+'].rename(columns={'pontuacao': 'media_R_plus'})
    df_p_minus = df_medias[df_medias['tipo_pergunta'] == 'P-'].rename(columns={'pontuacao': 'media_R_minus'})

    # Merge
    df_pares = pd.merge(
        df_p_plus, df_p_minus,
        on=['modelo', 'eixo', 'pair_id', 'temperatura', 'tendencia'],
        how='inner'
    )
    
    df_pares['diferenca_R'] = df_pares['media_R_plus'] - df_pares['media_R_minus']
    
    # IP Médio
    df_ip = df_pares.groupby(['modelo', 'temperatura', 'tendencia'])['diferenca_R'].mean().reset_index()
    df_ip = df_ip.rename(columns={'diferenca_R': 'indice_polarizacao'})
    
    return df_pares, df_ip

def get_model_size(model_name: str, params_db: dict) -> float | None:
    model_name_lower = str(model_name).lower()
    return params_db.get(model_name_lower)