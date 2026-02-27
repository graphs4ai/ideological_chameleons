import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from math import pi
from omegaconf import DictConfig
from .processing import get_model_size

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.max_open_warning': 0})

def save_fig(fig, name: str, cfg: DictConfig):
    if cfg.analysis.save_plots:
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.output_dir, "svg"), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.output_dir, "png"), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.output_dir, "pdf"), exist_ok=True)
        path_svg = os.path.join(cfg.paths.output_dir, f"svg/{name}.svg")
        path_png = os.path.join(cfg.paths.output_dir, f"png/{name}.png")
        path_pdf = os.path.join(cfg.paths.output_dir, f"pdf/{name}.pdf")
        fig.savefig(path_svg, format="svg", bbox_inches='tight')
        fig.savefig(path_png, format="png", bbox_inches='tight')
        fig.savefig(path_pdf, format="pdf", bbox_inches='tight')
        print(f"Figuras salvas em: {path_png}")

def plot_figure1_user_shifts_chameleon(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 1: User-Conditioned Shifts & Chameleon Index
    """
    # Prepare data - calculate mean and std IP per model and condition
    df_shifts = df_ip.groupby(['modelo', 'tendencia'])['indice_polarizacao'].agg(['mean', 'std']).reset_index()
    df_shifts.columns = ['modelo', 'tendencia', 'ip_mean', 'ip_std']
        
    # Pivot mean values
    df_pivot = df_shifts.pivot(index='modelo', columns='tendencia', values='ip_mean')
    df_pivot = df_pivot.reset_index()
    
    # Pivot std values
    df_pivot_std = df_shifts.pivot(index='modelo', columns='tendencia', values='ip_std')
    df_pivot_std = df_pivot_std.reset_index()
    df_pivot_std.columns = ['modelo', 'esquerda_std', 'neutro_std', 'direita_std']
    
    # Merge mean and std
    df_pivot = df_pivot.merge(df_pivot_std, on='modelo')
    
    # Calculate Chameleon Index (sum of absolute shifts from neutral)
    df_pivot['shift_left'] = abs(df_pivot['esquerda'] - df_pivot['neutro'])
    df_pivot['shift_right'] = abs(df_pivot['direita'] - df_pivot['neutro'])
    df_pivot['chameleon_index'] = df_pivot['shift_left'] + df_pivot['shift_right']
    
    # Sort by chameleon index for Panel B
    df_pivot_sorted = df_pivot.sort_values('chameleon_index', ascending=False)
    
    df_pivot_a = df_pivot.sort_values('neutro')
    y_pos = np.arange(len(df_pivot_a))
    colors_gradient = plt.cm.YlOrRd(df_pivot_sorted['chameleon_index'] / df_pivot_sorted['chameleon_index'].max())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 14), gridspec_kw={'width_ratios': [2, 1]})
    # Panel A
    for i, row in df_pivot_a.iterrows():
        x_values = [row['esquerda'], row['neutro'], row['direita']]
        ax1.plot(x_values, [y_pos[df_pivot_a.index.get_loc(i)]] * 3, color='gray', alpha=0.3, linewidth=1)
    ax1.errorbar(df_pivot_a['esquerda'], y_pos, xerr=df_pivot_a['esquerda_std'],
                fmt='o', markersize=15, color='#e74c3c', ecolor='#e74c3c',
                alpha=0.8, label='Left-Wing User', zorder=3, capsize=3, capthick=1.5)
    ax1.errorbar(df_pivot_a['neutro'], y_pos, xerr=df_pivot_a['neutro_std'],
                fmt='o', markersize=15, color='#95a5a6', ecolor='#95a5a6',
                alpha=0.8, label='Neutral User', zorder=3, capsize=3, capthick=1.5)
    ax1.errorbar(df_pivot_a['direita'], y_pos, xerr=df_pivot_a['direita_std'],
                fmt='o', markersize=15, color='#3498db', ecolor='#3498db',
                alpha=0.8, label='Right-Wing User', zorder=3, capsize=3, capthick=1.5)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_pivot_a['modelo'], fontsize=25)
    ax1.set_xlabel('Polarization Index (PI)', fontsize=26, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=26, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=21)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=26, ncol=3, frameon=False)
    ax1.grid(axis='x', alpha=0.3, linestyle=':')
    ax1.set_xlim(-4, 4)
    ax1.text(
        -0.40, 1.01, 'A',
        transform=ax1.transAxes,
        fontsize=28,
        fontweight='bold',
        va='top',
        ha='right'
    )
    ax2.text(
        -0.40, 1.01, 'B',
        transform=ax2.transAxes,
        fontsize=28,
        fontweight='bold',
        va='top',
        ha='right'
    )
    y_pos_b = np.arange(len(df_pivot_sorted))
    ax2.barh(y_pos_b, df_pivot_sorted['chameleon_index'], color=colors_gradient, alpha=0.85, height=0.7)
    ax2.set_yticks(y_pos_b)
    ax2.set_yticklabels(df_pivot_sorted['modelo'], fontsize=25)
    ax2.set_xlabel('Chameleon Index (CI)', fontsize=26, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=21)
    ax2.grid(axis='x', alpha=0.3, linestyle=':')
    ax2.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, "figure1_user_shifts_and_chameleon", cfg)
    plt.close()

def plot_figure2_topic_variation(df_pares: pd.DataFrame, cfg: DictConfig):
    """
    Figure 2: Topic-Level Variation
    Heatmap showing Chameleon Index per model and topic (axis).
    """
    topic_translation = {
        'Políticas Sociais': 'Welfare',
        'Segurança Pública': 'Security',
        'Economia': 'Economy',
        'Meio Ambiente': 'Environment',
        'Educação e Cultura': 'Education and Culture',
        'Corrupção e Justiça': 'Corruption and Justice',
        'Instituições Democráticas': 'Democratic Institutions'
    }
    
    df_topic = df_pares.groupby(['modelo', 'eixo', 'tendencia'])['diferenca_R'].mean().reset_index()
    
    # Pivot to get neutral, left, right for each model-topic combination
    df_topic_pivot = df_topic.pivot_table(
        index=['modelo', 'eixo'], 
        columns='tendencia', 
        values='diferenca_R'
    ).reset_index()
    
    # Calculate Chameleon Index per topic
    df_topic_pivot['shift_left'] = abs(df_topic_pivot['esquerda'] - df_topic_pivot['neutro'])
    df_topic_pivot['shift_right'] = abs(df_topic_pivot['direita'] - df_topic_pivot['neutro'])
    df_topic_pivot['chameleon_index'] = df_topic_pivot['shift_left'] + df_topic_pivot['shift_right']
    
    # Create heatmap matrix
    heatmap_data = df_topic_pivot.pivot(
        index='modelo', 
        columns='eixo', 
        values='chameleon_index'
    )
    
    # Translate column names to English
    heatmap_data.columns = [topic_translation.get(col, col) for col in heatmap_data.columns]
    
    # Sort rows by average chameleon index across all topics
    heatmap_data['avg'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values('avg', ascending=True).drop('avg', axis=1)
    
    # Sort columns (topics) by average chameleon index across all models
    topic_avg = heatmap_data.mean(axis=0).sort_values(ascending=True)
    heatmap_data = heatmap_data[topic_avg.index]
    
    # Create figure (size adjusted for all models)
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Chameleon Index (CI)'}, 
                linewidths=0.5, linecolor='white', vmin=0, vmax=8)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Chameleon Index (CI)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
        
    # Labels and title
    ax.set_xlabel('Topic', fontsize=19, fontweight='bold')
    ax.set_ylabel('Model', fontsize=19, fontweight='bold')
    ax.tick_params(axis='y', labelsize=19)
    ax.tick_params(axis='x', labelsize=19) 
    for text in ax.texts:
        text.set_fontsize(15)
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    save_fig(fig, "figure2_topic_variation", cfg)
    plt.close()

def plot_figure3_likert_distribution(df_validos: pd.DataFrame, cfg: DictConfig):
    """
    Figure 3: Response Distribution (Likert Scale)
    Grouped histogram showing response counts across Likert scale for 3 user types.
    """
    # Define Likert scale mapping
    likert_labels = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
    likert_values = [-2, -1, 0, 1, 2]
    
    # Count responses by tendency and score
    df_counts = df_validos.groupby(['tendencia', 'pontuacao']).size().reset_index(name='count')
    
    # Pivot for easier plotting
    df_pivot = df_counts.pivot(index='pontuacao', columns='tendencia', values='count').fillna(0)
    
    # Reindex to ensure all Likert values are present
    df_pivot = df_pivot.reindex(likert_values, fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define bar width and positions
    bar_width = 0.25
    x_pos = np.arange(len(likert_values))
    
    # Define colors and labels
    colors = {'neutro': '#95a5a6', 'esquerda': '#e74c3c', 'direita': '#3498db'}
    labels = {'neutro': 'Neutral User', 'esquerda': 'Left-Wing User', 'direita': 'Right-Wing User'}
    
    # Plot bars for each tendency
    tendencies = ['neutro', 'esquerda', 'direita']
    for i, tend in enumerate(tendencies):
        if tend in df_pivot.columns:
            offset = (i - 1) * bar_width
            ax.bar(x_pos + offset, df_pivot[tend], bar_width, 
                  label=labels[tend], color=colors[tend], alpha=0.85)
    
    # Labels and formatting
    ax.set_xlabel('Likert Scale Response', fontsize=17, fontweight='bold')
    ax.set_ylabel('Response Count', fontsize=17, fontweight='bold')
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(likert_labels)
    #Tamanho do eixo x
    ax.tick_params(axis='x', labelsize=19)
    # Add legend
    ax.legend(loc='upper left', fontsize=15)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    save_fig(fig, "figure3_likert_distribution", cfg)
    plt.close()

def plot_figure4_temperature_robustness(df_ip: pd.DataFrame, cfg: DictConfig):  
    """
    Figure 4: Robustness Across Temperatures (Heatmap Version)
    Heatmap showing IP values for each model at different temperatures.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    
    # Filter data for neutral tendency only
    df_temp = df_ip[
        (df_ip['tendencia'] == 'neutro') & 
        (df_ip['indice_polarizacao'].notna())
    ].copy()
    
    if df_temp.empty:
        print("Warning: No valid data found for temperature robustness analysis.")
        return

    # Calculate mean IP per model and temperature
    df_temp_agg = df_temp.groupby(['modelo', 'temperatura']).agg({
        'indice_polarizacao': 'mean'
    }).reset_index()
    df_temp_agg.columns = ['Model', 'Temperature', 'IP_mean']
    
    # Create pivot table for heatmap (Temperature as rows, Models as columns)
    heatmap_data = df_temp_agg.pivot(
        index='Temperature',
        columns='Model',
        values='IP_mean'
    )
    
    # Sort models by average IP across all temperatures, but put GPT-5 at the end
    avg_ip_per_model = heatmap_data.mean(axis=0).sort_values(ascending=False)
    
    # Separate GPT-5 (if exists) and other models
    gpt5_models = [col for col in avg_ip_per_model.index if 'gpt-5-nano' in col.lower()]
    other_models = [col for col in avg_ip_per_model.index if col not in gpt5_models]
    
    # Reorder: other models sorted by IP, then GPT-5 models at the end
    new_order = other_models + gpt5_models
    heatmap_data = heatmap_data[new_order]
    
    # Define bins for discrete color mapping
    bins = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    
    # Create discrete colormap
    cmap = plt.cm.RdBu_r
    norm = BoundaryNorm(bins, cmap.N)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create heatmap with discrete colormap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                norm=norm,
                ax=ax, 
                cbar_kws={'label': 'Polarization Index (PI)', 'ticks': bins, 'pad': 0.01},
                linewidths=0.5, 
                linecolor='white')
    
    # Format colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Polarization Index (PI)', fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=15)
    
    # Labels and formatting
    ax.set_xlabel('Model', fontsize=15, fontweight='bold')
    ax.set_ylabel('Temperature', fontsize=15, fontweight='bold')
    
    # Adjust tick label sizes
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=13)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust annotation font size
    for text in ax.texts:
        text.set_fontsize(14)
    
    plt.tight_layout()
    save_fig(fig, "figure4_temperature_robustness", cfg)
    plt.close()
