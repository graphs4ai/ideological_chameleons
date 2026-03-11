import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

# Garante que o python encontre o módulo src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis import processing, plotting, statistics

@hydra.main(version_base=None, config_path="conf", config_name="analysis_config")
def main(cfg: DictConfig) -> None:
    
    plotting.setup_style()

    try:
        df_validos = processing.carregar_e_processar_dados(cfg)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{cfg.paths.input_file}' não encontrado.")
        return

    df_pares, df_ip = processing.calcular_indice_polarizacao(df_validos)
    
    if df_ip.empty:
        print("Aviso: Nenhum dado de polarização calculado.")
        return

    plotting.plot_figure1_user_shifts_chameleon(df_ip, cfg)
    plotting.plot_figure2_topic_variation(df_pares, cfg)
    plotting.plot_figure3_likert_distribution(df_validos, cfg)
    plotting.plot_figure4_temperature_robustness(df_ip, cfg)
    plotting.plot_figure4_2_temperature_deltas(df_ip, cfg)
    plotting.plot_figure4_1_temperature_mixed(df_ip, cfg)
    plotting.plot_figure4_1_1_temperature_mixed_aligned(df_ip, cfg)
    plotting.plot_figure4_3_temperature_ipi(df_ip, cfg)
    plotting.plot_figure4_4_temperature_ipi_aligned(df_ip, cfg)
    plotting.plot_figure4_5_temperature_deltas_aligned(df_ip, cfg)
    plotting.plot_figure5_topic_dot_panel(df_pares, cfg)
    plotting.plot_figure5_5_topic_dot_separate(df_pares, cfg)
    plotting.plot_figure2_and_5_combined(df_pares, cfg)
    plotting.plot_figure6_size_vs_chameleon(df_ip, cfg)
    plotting.plot_figure6_1_size_vs_ipi_nocontext(df_ip, cfg)
    plotting.plot_figure7_agreement_heatmap(df_validos, cfg)
    plotting.plot_figure8_swing_asymmetry(df_ip, cfg)
    statistics.std_v(cfg)



if __name__ == "__main__":
    main()