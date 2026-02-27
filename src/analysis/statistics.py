import pandas as pd
from omegaconf import DictConfig

def std_v(cfg: DictConfig) -> pd.DataFrame:
    likert_map = dict(cfg.analysis.likert_map)
    df_resultados = pd.read_csv(cfg.paths.input_file)
    df_resultados['pontuacao'] = df_resultados['resposta_raw'].map(likert_map)

    # Filtrar inválidos
    df_validos = df_resultados.dropna(subset=['pontuacao']).copy()
    df_validos['pontuacao'] = df_validos['pontuacao'].astype(int)


    df_medias = df_validos.groupby(
    ['modelo', 'eixo', 'pair_id', 'tipo_pergunta', 'temperatura', 'tendencia', 'repeticao']
    )['pontuacao'].mean().reset_index()

    df_p_plus = df_medias[df_medias['tipo_pergunta'] == 'P+'] \
        .rename(columns={'pontuacao': 'media_R_plus'})

    df_p_minus = df_medias[df_medias['tipo_pergunta'] == 'P-'] \
        .rename(columns={'pontuacao': 'media_R_minus'})

    df_pares = pd.merge(
        df_p_plus,
        df_p_minus,
        on=['modelo', 'eixo', 'pair_id', 'temperatura', 'tendencia', 'repeticao'],
        how='inner',
        suffixes=('_plus', '_minus')
    )

    df_pares['diferenca_R'] = df_pares['media_R_plus'] - df_pares['media_R_minus']

    df_ip = df_pares.groupby(['modelo', 'temperatura', 'tendencia', 'repeticao'])['diferenca_R'].mean().reset_index()
    df_ip = df_ip.rename(columns={'diferenca_R': 'indice_polarizacao'})

    df_neutro = df_ip[df_ip['tendencia'] == 'neutro']

    std_ip_repeticao = (
        df_neutro
        .groupby(['modelo', 'temperatura'])['indice_polarizacao']
        .std()
        .reset_index()
        .rename(columns={'indice_polarizacao': 'std_ip_repeticao'})
    )

    std_ip_repeticao = std_ip_repeticao.sort_values(
        by='std_ip_repeticao', ascending=False
    )

    std_ip_repeticao.to_csv('std_ip_repeticao.txt', index=False, sep='\t')
    return std_ip_repeticao

    