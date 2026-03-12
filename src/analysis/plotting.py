import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from math import pi
from scipy import stats
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
                alpha=0.8, label='No-Context User', zorder=3, capsize=3, capthick=1.5)
    ax1.errorbar(df_pivot_a['direita'], y_pos, xerr=df_pivot_a['direita_std'],
                fmt='o', markersize=15, color='#3498db', ecolor='#3498db',
                alpha=0.8, label='Right-Wing User', zorder=3, capsize=3, capthick=1.5)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_pivot_a['modelo'], fontsize=25)
    ax1.set_xlabel('Ideological Position Index (IPI)', fontsize=26, fontweight='bold')
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


def plot_figure2_and_5_combined(df_pares: pd.DataFrame, cfg: DictConfig):
    """
    Figure 2 and 5 Combined: Topic-Level Analysis
    A: Chameleon Index per Model and Topic (Heatmap)
    B: Topic-Level IPI across User Types (Dot Plot)
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
    
    # Create figure with two side-by-side panels
    fig = plt.figure(figsize=(32, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.40)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # ==================== PANEL A: Heatmap (Figure 2) ====================
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
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax1, cbar_kws={'label': 'Chameleon Index (CI)'}, 
                linewidths=0.5, linecolor='white', vmin=0, vmax=8)
    cbar = ax1.collections[0].colorbar
    cbar.set_label('Chameleon Index (CI)', fontsize=22, fontweight='bold')
    cbar.ax.tick_params(labelsize=18)
        
    # Labels
    ax1.set_xlabel('Topic', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=24, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=22)
    ax1.tick_params(axis='x', labelsize=22) 
    for text in ax1.texts:
        text.set_fontsize(18)
    # Rotate x-axis labels
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Add panel label
    ax1.text(-0.68, 1.02, 'A', transform=ax1.transAxes, fontsize=32,
             fontweight='bold', va='top', ha='right')
    
    # ==================== PANEL B: Dot Plot (Figure 5) ====================
    # Mean IPI per topic and tendency (averaged across models)
    df_topic_mean = df_pares.groupby(['eixo', 'tendencia'])['diferenca_R'].mean().reset_index()
    df_topic_std = df_pares.groupby(['eixo', 'tendencia'])['diferenca_R'].std().reset_index()
    df_topic_std.columns = ['eixo', 'tendencia', 'std']
    df_topic_combined = df_topic_mean.merge(df_topic_std, on=['eixo', 'tendencia'])
    
    # Translate topic names
    df_topic_combined['topic'] = df_topic_combined['eixo'].map(topic_translation)
    
    # Sort topics by the spread (right - left mean) for visual clarity
    topic_order_df = df_topic_combined.pivot(index='topic', columns='tendencia', values='diferenca_R').reset_index()
    topic_order_df['spread'] = abs(topic_order_df['direita'] - topic_order_df['esquerda'])
    topic_order = topic_order_df.sort_values('spread', ascending=True)['topic'].tolist()
    
    y_pos = np.arange(len(topic_order))
    colors = {'neutro': '#95a5a6', 'esquerda': '#e74c3c', 'direita': '#3498db'}
    labels = {'neutro': 'No-Context User', 'esquerda': 'Left-Wing User', 'direita': 'Right-Wing User'}
    markers = {'neutro': 's', 'esquerda': 'o', 'direita': 'D'}
    
    for tend in ['esquerda', 'neutro', 'direita']:
        subset = df_topic_combined[df_topic_combined['tendencia'] == tend].copy()
        subset['y'] = subset['topic'].map({t: i for i, t in enumerate(topic_order)})
        subset = subset.sort_values('y')
        ax2.errorbar(subset['diferenca_R'], subset['y'], xerr=subset['std'],
                    fmt=markers[tend], markersize=16, color=colors[tend],
                    ecolor=colors[tend], alpha=0.8, label=labels[tend],
                    capsize=5, capthick=2, zorder=3, linewidth=0)
    
    # Connect left-right with gray lines per topic
    for i, topic in enumerate(topic_order):
        vals = df_topic_combined[df_topic_combined['topic'] == topic]
        left_val = vals[vals['tendencia'] == 'esquerda']['diferenca_R'].values
        right_val = vals[vals['tendencia'] == 'direita']['diferenca_R'].values
        if len(left_val) > 0 and len(right_val) > 0:
            ax2.plot([left_val[0], right_val[0]], [i, i], color='gray', alpha=0.3, linewidth=2, zorder=1)
    
    ax2.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(topic_order, fontsize=22)
    ax2.set_xlabel('Ideological Position Index (IPI)', fontsize=24, fontweight='bold')
    ax2.set_ylabel('Topic', fontsize=24, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=20)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=20, ncol=3, frameon=False)
    ax2.grid(axis='x', alpha=0.3, linestyle=':')
    
    # Add panel label
    ax2.text(-0.28, 1.02, 'B', transform=ax2.transAxes, fontsize=32,
             fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    save_fig(fig, "figure2_and_5_combined", cfg)
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
    labels = {'neutro': 'No-Context User', 'esquerda': 'Left-Wing User', 'direita': 'Right-Wing User'}
    
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
                cbar_kws={'label': 'Ideological Position Index (IPI)', 'ticks': bins, 'pad': 0.01},
                linewidths=0.5, 
                linecolor='white')
    
    # Format colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Ideological Position Index (IPI)', fontsize=15, fontweight='bold')
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


def _prepare_temperature_data(df_ip: pd.DataFrame):
    """Helper: prepare pivoted temperature data with IPI per tendency and deltas."""
    df_temp_all = df_ip[df_ip['indice_polarizacao'].notna()].copy()
    if df_temp_all.empty:
        return None
    df_agg = df_temp_all.groupby(['modelo', 'temperatura', 'tendencia'])['indice_polarizacao'].mean().reset_index()
    df_pivot = df_agg.pivot_table(
        index=['modelo', 'temperatura'],
        columns='tendencia',
        values='indice_polarizacao'
    ).reset_index()
    df_pivot['delta_left'] = abs(df_pivot['esquerda'] - df_pivot['neutro'])
    df_pivot['delta_right'] = abs(df_pivot['direita'] - df_pivot['neutro'])
    return df_pivot


def _build_temp_heatmap(df, value_col):
    """Helper: pivot temperature data into heatmap matrix, sorted by avg, GPT-5 last."""
    hm = df.pivot(index='temperatura', columns='modelo', values=value_col)
    avg = hm.mean(axis=0).sort_values(ascending=False)
    gpt5 = [c for c in avg.index if 'gpt-5-nano' in c.lower()]
    others = [c for c in avg.index if c not in gpt5]
    return hm[others + gpt5]


def _draw_temp_heatmap(ax, data, cmap, label, fmt='.2f', vmin=None, vmax=None,
                       norm=None, cbar_ticks=None, annot_size=13, label_size=13,
                       show_xlabel=False, show_xticklabels=True):
    """Helper: draw a single temperature heatmap on a given axes."""
    kws = {'label': label, 'pad': 0.01}
    if cbar_ticks is not None:
        kws['ticks'] = cbar_ticks
    hm_kwargs = dict(annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar_kws=kws,
                     linewidths=0.5, linecolor='white')
    if norm is not None:
        hm_kwargs['norm'] = norm
    if vmin is not None:
        hm_kwargs['vmin'] = vmin
    if vmax is not None:
        hm_kwargs['vmax'] = vmax
    sns.heatmap(data, **hm_kwargs)
    cbar = ax.collections[0].colorbar
    cbar.set_label(label, fontsize=label_size, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    ax.set_xlabel('Model' if show_xlabel else '', fontsize=14 if show_xlabel else 1, fontweight='bold')
    ax.set_ylabel('Temperature', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=12)
    if show_xticklabels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.set_xticklabels([])
    for t in ax.texts:
        t.set_fontsize(annot_size)


def plot_figure4_1_temperature_mixed(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 4.1: Temperature Robustness (Mixed)
    A: Base IPI (neutral), B: |ΔIPI left|, C: |ΔIPI right|.
    """
    from matplotlib.colors import BoundaryNorm
    df_pivot = _prepare_temperature_data(df_ip)
    if df_pivot is None:
        print("Warning: No valid data for figure 4.1."); return

    hm_base = _build_temp_heatmap(df_pivot, 'neutro')
    hm_left = _build_temp_heatmap(df_pivot, 'delta_left')
    hm_right = _build_temp_heatmap(df_pivot, 'delta_right')

    fig, axes = plt.subplots(3, 1, figsize=(16, 20), constrained_layout=True)
    bins = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    norm = BoundaryNorm(bins, plt.cm.RdBu_r.N)
    vmax_lr = max(hm_left.max().max(), hm_right.max().max())

    _draw_temp_heatmap(axes[0], hm_base, plt.cm.RdBu_r, 'Ideological Position Index (IPI)',
                       norm=norm, cbar_ticks=bins)
    axes[0].text(-0.05, 1.05, 'A', transform=axes[0].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    _draw_temp_heatmap(axes[1], hm_left, 'Reds', '|ΔIPI left|', vmin=0, vmax=vmax_lr)
    axes[1].text(-0.05, 1.05, 'B', transform=axes[1].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    _draw_temp_heatmap(axes[2], hm_right, 'Blues', '|ΔIPI right|', vmin=0, vmax=vmax_lr,
                       show_xlabel=True)
    axes[2].text(-0.05, 1.05, 'C', transform=axes[2].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    save_fig(fig, "figure4_1_temperature_mixed", cfg)
    plt.close()


def plot_figure4_1_1_temperature_mixed_aligned(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 4.1.1: Temperature Robustness (Mixed - Aligned)
    A: Base IPI (neutral), B: |ΔIPI left|, C: |ΔIPI right|.
    Model names shown only at the bottom.
    """
    from matplotlib.colors import BoundaryNorm
    df_pivot = _prepare_temperature_data(df_ip)
    if df_pivot is None:
        print("Warning: No valid data for figure 4.1.1."); return

    hm_base = _build_temp_heatmap(df_pivot, 'neutro')
    hm_left = _build_temp_heatmap(df_pivot, 'delta_left')
    hm_right = _build_temp_heatmap(df_pivot, 'delta_right')

    # Ensure same column order across all
    cols = hm_base.columns.tolist()
    hm_left = hm_left[cols]
    hm_right = hm_right[cols]

    bins = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    norm = BoundaryNorm(bins, plt.cm.RdBu_r.N)
    vmax_lr = max(hm_left.max().max(), hm_right.max().max())

    fig, axes = plt.subplots(3, 1, figsize=(16, 20),
                             gridspec_kw={'hspace': 0.08})

    # Panel A: Base IPI (neutral) - no x-axis labels
    _draw_temp_heatmap(axes[0], hm_base, plt.cm.RdBu_r, 'Ideological Position Index (IPI)',
                       norm=norm, cbar_ticks=bins, show_xticklabels=False)
    axes[0].text(-0.05, 1.05, 'A', transform=axes[0].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    # Panel B: |ΔIPI left| - no x-axis labels
    _draw_temp_heatmap(axes[1], hm_left, 'Reds', '|ΔIPI left|', vmin=0, vmax=vmax_lr,
                       show_xticklabels=False)
    axes[1].text(-0.05, 1.05, 'B', transform=axes[1].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    # Panel C: |ΔIPI right| - show x-axis labels
    _draw_temp_heatmap(axes[2], hm_right, 'Blues', '|ΔIPI right|', vmin=0, vmax=vmax_lr,
                       show_xlabel=True, show_xticklabels=True)
    axes[2].text(-0.05, 1.05, 'C', transform=axes[2].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    fig.subplots_adjust(bottom=0.12)
    save_fig(fig, "figure4_1_1_temperature_mixed_aligned", cfg)
    plt.close()


def plot_figure4_2_temperature_deltas(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 4.2: Temperature Robustness — All Deltas
    A: |ΔIPI left|, B: |ΔIPI right|.
    """
    df_pivot = _prepare_temperature_data(df_ip)
    if df_pivot is None:
        print("Warning: No valid data for figure 4.2."); return

    hm_left = _build_temp_heatmap(df_pivot, 'delta_left')
    hm_right = _build_temp_heatmap(df_pivot, 'delta_right')
    vmax = max(hm_left.max().max(), hm_right.max().max())

    fig, axes = plt.subplots(2, 1, figsize=(16, 14), constrained_layout=True)

    _draw_temp_heatmap(axes[0], hm_left, 'Reds', '|ΔIPI left|', vmin=0, vmax=vmax)
    axes[0].text(-0.05, 1.05, 'A', transform=axes[0].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    _draw_temp_heatmap(axes[1], hm_right, 'Blues', '|ΔIPI right|', vmin=0, vmax=vmax,
                       show_xlabel=True)
    axes[1].text(-0.05, 1.05, 'B', transform=axes[1].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    save_fig(fig, "figure4_2_temperature_deltas", cfg)
    plt.close()


def plot_figure4_3_temperature_ipi(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 4.3: Temperature Robustness — All IPI
    A: IPI Left-Wing, B: IPI No-Context, C: IPI Right-Wing.
    """
    from matplotlib.colors import BoundaryNorm
    df_pivot = _prepare_temperature_data(df_ip)
    if df_pivot is None:
        print("Warning: No valid data for figure 4.3."); return

    hm_left = _build_temp_heatmap(df_pivot, 'esquerda')
    hm_neutral = _build_temp_heatmap(df_pivot, 'neutro')
    hm_right = _build_temp_heatmap(df_pivot, 'direita')

    bins = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    norm = BoundaryNorm(bins, plt.cm.RdBu_r.N)

    fig, axes = plt.subplots(3, 1, figsize=(16, 20), constrained_layout=True)

    _draw_temp_heatmap(axes[0], hm_left, plt.cm.RdBu_r, 'IPI (Left-Wing User)',
                       norm=norm, cbar_ticks=bins)
    axes[0].text(-0.05, 1.05, 'A', transform=axes[0].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    _draw_temp_heatmap(axes[1], hm_neutral, plt.cm.RdBu_r, 'IPI (No-Context User)',
                       norm=norm, cbar_ticks=bins)
    axes[1].text(-0.05, 1.05, 'B', transform=axes[1].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    _draw_temp_heatmap(axes[2], hm_right, plt.cm.RdBu_r, 'IPI (Right-Wing User)',
                       norm=norm, cbar_ticks=bins, show_xlabel=True)
    axes[2].text(-0.05, 1.05, 'C', transform=axes[2].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    save_fig(fig, "figure4_3_temperature_ipi", cfg)
    plt.close()


def plot_figure4_4_temperature_ipi_aligned(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 4.4: Temperature IPI — Aligned models (names shown once at bottom).
    A: IPI Left-Wing, B: IPI No-Context, C: IPI Right-Wing.
    """
    from matplotlib.colors import BoundaryNorm
    df_pivot = _prepare_temperature_data(df_ip)
    if df_pivot is None:
        print("Warning: No valid data for figure 4.4."); return

    hm_left = _build_temp_heatmap(df_pivot, 'esquerda')
    hm_neutral = _build_temp_heatmap(df_pivot, 'neutro')
    hm_right = _build_temp_heatmap(df_pivot, 'direita')

    # Ensure same column order across all
    cols = hm_neutral.columns.tolist()
    hm_left = hm_left[cols]
    hm_right = hm_right[cols]

    bins = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    norm = BoundaryNorm(bins, plt.cm.RdBu_r.N)

    fig, axes = plt.subplots(3, 1, figsize=(16, 20),
                             gridspec_kw={'hspace': 0.08})

    panels = [
        (axes[0], hm_left, 'IPI (Left-Wing User)', 'A', False),
        (axes[1], hm_neutral, 'IPI (No-Context User)', 'B', False),
        (axes[2], hm_right, 'IPI (Right-Wing User)', 'C', True),
    ]
    for ax, data, label, letter, show_x in panels:
        _draw_temp_heatmap(ax, data, plt.cm.RdBu_r, label, norm=norm,
                           cbar_ticks=bins, show_xlabel=show_x,
                           show_xticklabels=show_x)
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes, fontsize=22,
                fontweight='bold', va='top', ha='right')

    fig.subplots_adjust(bottom=0.12)
    save_fig(fig, "figure4_4_temperature_ipi_aligned", cfg)
    plt.close()


def plot_figure4_5_temperature_deltas_aligned(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 4.5: Temperature Deltas — Aligned models (names shown once at bottom).
    A: |ΔIPI left|, B: |ΔIPI right|.
    """
    df_pivot = _prepare_temperature_data(df_ip)
    if df_pivot is None:
        print("Warning: No valid data for figure 4.5."); return

    hm_left = _build_temp_heatmap(df_pivot, 'delta_left')
    hm_right = _build_temp_heatmap(df_pivot, 'delta_right')

    cols = hm_left.columns.tolist()
    hm_right = hm_right[cols]
    vmax = max(hm_left.max().max(), hm_right.max().max())

    fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                             gridspec_kw={'hspace': 0.08})

    _draw_temp_heatmap(axes[0], hm_left, 'Reds', '|ΔIPI left|',
                       vmin=0, vmax=vmax, show_xticklabels=False)
    axes[0].text(-0.05, 1.05, 'A', transform=axes[0].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    _draw_temp_heatmap(axes[1], hm_right, 'Blues', '|ΔIPI right|',
                       vmin=0, vmax=vmax, show_xlabel=True, show_xticklabels=True)
    axes[1].text(-0.05, 1.05, 'B', transform=axes[1].transAxes, fontsize=22,
                 fontweight='bold', va='top', ha='right')

    fig.subplots_adjust(bottom=0.12)
    save_fig(fig, "figure4_5_temperature_deltas_aligned", cfg)
    plt.close()


def plot_figure5_topic_dot_panel(df_pares: pd.DataFrame, cfg: DictConfig):
    """
    Figure 5: Topic-Level Dot Plot
    Y-axis: 7 topics. X-axis: IPI.
    Shows mean (across models) IPI for No-Context, Left, and Right users per topic.
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
    
    # Mean IPI per topic and tendency (averaged across models)
    df_topic = df_pares.groupby(['eixo', 'tendencia'])['diferenca_R'].mean().reset_index()
    df_topic_std = df_pares.groupby(['eixo', 'tendencia'])['diferenca_R'].std().reset_index()
    df_topic_std.columns = ['eixo', 'tendencia', 'std']
    df_topic = df_topic.merge(df_topic_std, on=['eixo', 'tendencia'])
    
    # Translate topic names
    df_topic['topic'] = df_topic['eixo'].map(topic_translation)
    
    # Sort topics by the spread (right - left mean) for visual clarity
    topic_order_df = df_topic.pivot(index='topic', columns='tendencia', values='diferenca_R').reset_index()
    topic_order_df['spread'] = abs(topic_order_df['direita'] - topic_order_df['esquerda'])
    topic_order = topic_order_df.sort_values('spread', ascending=True)['topic'].tolist()
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    y_pos = np.arange(len(topic_order))
    colors = {'neutro': '#95a5a6', 'esquerda': '#e74c3c', 'direita': '#3498db'}
    labels = {'neutro': 'No-Context User', 'esquerda': 'Left-Wing User', 'direita': 'Right-Wing User'}
    markers = {'neutro': 's', 'esquerda': 'o', 'direita': 'D'}
    
    for tend in ['esquerda', 'neutro', 'direita']:
        subset = df_topic[df_topic['tendencia'] == tend].copy()
        subset['y'] = subset['topic'].map({t: i for i, t in enumerate(topic_order)})
        subset = subset.sort_values('y')
        ax.errorbar(subset['diferenca_R'], subset['y'], xerr=subset['std'],
                    fmt=markers[tend], markersize=14, color=colors[tend],
                    ecolor=colors[tend], alpha=0.8, label=labels[tend],
                    capsize=4, capthick=1.5, zorder=3, linewidth=0)
    
    # Connect left-right with gray lines per topic
    for i, topic in enumerate(topic_order):
        vals = df_topic[df_topic['topic'] == topic]
        left_val = vals[vals['tendencia'] == 'esquerda']['diferenca_R'].values
        right_val = vals[vals['tendencia'] == 'direita']['diferenca_R'].values
        if len(left_val) > 0 and len(right_val) > 0:
            ax.plot([left_val[0], right_val[0]], [i, i], color='gray', alpha=0.3, linewidth=1.5, zorder=1)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topic_order, fontsize=18)
    ax.set_xlabel('Ideological Position Index (IPI)', fontsize=19, fontweight='bold')
    ax.set_ylabel('Topic', fontsize=19, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=17, ncol=3, frameon=False)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    save_fig(fig, "figure5_topic_dot_panel", cfg)
    plt.close()


def plot_figure5_5_topic_dot_separate(df_pares: pd.DataFrame, cfg: DictConfig):
    """
    Figure 5.5: Topic-Level Dot Plot (Separate Panels)
    Three side-by-side panels showing IPI per topic for each user tendency separately.
    A: Left-Wing User, B: No-Context User, C: Right-Wing User.
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
    
    # Mean IPI per topic and tendency (averaged across models)
    df_topic = df_pares.groupby(['eixo', 'tendencia'])['diferenca_R'].mean().reset_index()
    df_topic_std = df_pares.groupby(['eixo', 'tendencia'])['diferenca_R'].std().reset_index()
    df_topic_std.columns = ['eixo', 'tendencia', 'std']
    df_topic = df_topic.merge(df_topic_std, on=['eixo', 'tendencia'])
    
    # Translate topic names
    df_topic['topic'] = df_topic['eixo'].map(topic_translation)
    
    # Sort topics by the spread (right - left mean) for visual clarity
    topic_order_df = df_topic.pivot(index='topic', columns='tendencia', values='diferenca_R').reset_index()
    topic_order_df['spread'] = abs(topic_order_df['direita'] - topic_order_df['esquerda'])
    topic_order = topic_order_df.sort_values('spread', ascending=True)['topic'].tolist()
    
    # Create figure with 3 side-by-side panels
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), sharey=True)
    
    y_pos = np.arange(len(topic_order))
    colors = {'neutro': '#95a5a6', 'esquerda': '#e74c3c', 'direita': '#3498db'}
    labels = {'neutro': 'No-Context User', 'esquerda': 'Left-Wing User', 'direita': 'Right-Wing User'}
    markers = {'neutro': 's', 'esquerda': 'o', 'direita': 'D'}
    tendencies = ['esquerda', 'neutro', 'direita']
    panel_labels = ['A', 'B', 'C']
    
    # Determine global x-axis limits
    all_values = []
    for tend in tendencies:
        subset = df_topic[df_topic['tendencia'] == tend].copy()
        all_values.extend(subset['diferenca_R'].values)
        all_values.extend((subset['diferenca_R'] - subset['std']).values)
        all_values.extend((subset['diferenca_R'] + subset['std']).values)
    x_min, x_max = min(all_values), max(all_values)
    x_range = x_max - x_min
    x_lim = [x_min - 0.1 * x_range, x_max + 0.1 * x_range]
    
    # Plot each tendency in its own panel
    for idx, tend in enumerate(tendencies):
        ax = axes[idx]
        subset = df_topic[df_topic['tendencia'] == tend].copy()
        subset['y'] = subset['topic'].map({t: i for i, t in enumerate(topic_order)})
        subset = subset.sort_values('y')
        
        ax.errorbar(subset['diferenca_R'], subset['y'], xerr=subset['std'],
                    fmt=markers[tend], markersize=14, color=colors[tend],
                    ecolor=colors[tend], alpha=0.8,
                    capsize=4, capthick=1.5, zorder=3, linewidth=0)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_yticks(y_pos)
        if idx == 0:
            ax.set_yticklabels(topic_order, fontsize=16)
            ax.set_ylabel('Topic', fontsize=18, fontweight='bold')
        ax.set_xlabel('Ideological Position Index (IPI)', fontsize=18, fontweight='bold')
        ax.tick_params(axis='x', labelsize=14)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        ax.set_xlim(x_lim)
        
        # Add panel label
        ax.text(-0.15, 1.05, panel_labels[idx], transform=ax.transAxes, fontsize=24,
                fontweight='bold', va='top', ha='right')
        
        # Add title with tendency name
        ax.set_title(labels[tend], fontsize=19, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_fig(fig, "figure5_5_topic_dot_separate", cfg)
    plt.close()


def plot_figure6_size_vs_chameleon(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 6 (Supplementary): Scatter plot of Model Size (B params) vs Chameleon Index.
    Shows that size does not dictate sycophancy level.
    """
    from scipy import stats
    
    # Known approximate parameter counts (in billions)
    params_db = {
        'google/gemma-3-4b-it': 4,
        'google/gemma-3-12b-it': 12,
        'google/gemma-3-27b-it': 27,
        'mistralai/mixtral-8x7b-instruct-v0.1': 47,  # MoE total
        'mistralai/mistral-small-3.2-24b-instruct-2506': 24,
        'openai/gpt-oss-120b': 120,
        'openai/gpt-oss-20b': 20,
        'qwen/qwen3-14b': 14,
        'qwen/qwen3-32b': 32,
        'qwen/qwen3-235b-a22b-instruct-2507': 235,  # MoE total
        'meta-llama/meta-llama-3.1-8b-instruct': 8,
        'meta-llama/meta-llama-3.1-70b-instruct': 70,
        'meta-llama/llama-4-scout-17b-16e-instruct': 109,  # MoE total
        'deepseek-ai/deepseek-v3.2': 671,  # MoE total
        'microsoft/phi-4': 14,
        'nvidia/nvidia-nemotron-nano-12b-v2-vl': 12,
    }
    
    # Calculate Chameleon Index per model
    df_shifts = df_ip.groupby(['modelo', 'tendencia'])['indice_polarizacao'].mean().reset_index()
    df_pivot = df_shifts.pivot(index='modelo', columns='tendencia', values='indice_polarizacao').reset_index()
    
    if 'neutro' not in df_pivot.columns:
        print("Warning: 'neutro' tendency not found for size vs chameleon plot.")
        return
    
    df_pivot['shift_left'] = abs(df_pivot['esquerda'] - df_pivot['neutro'])
    df_pivot['shift_right'] = abs(df_pivot['direita'] - df_pivot['neutro'])
    df_pivot['chameleon_index'] = df_pivot['shift_left'] + df_pivot['shift_right']
    
    # Map model sizes
    df_pivot['size_b'] = df_pivot['modelo'].apply(lambda x: get_model_size(x, params_db))
    
    # Drop models without known size (e.g., closed-source like gpt-5-nano)
    df_plot = df_pivot.dropna(subset=['size_b']).copy()
    
    if df_plot.empty:
        print("Warning: No models with known sizes for scatter plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(df_plot['size_b'], df_plot['chameleon_index'],
               s=120, color='#2c3e50', alpha=0.8, edgecolors='white', linewidth=1.2, zorder=3)
    
    # Annotate each point
    for _, row in df_plot.iterrows():
        short_name = row['modelo'].split('/')[-1] if '/' in row['modelo'] else row['modelo']
        ax.annotate(short_name, (row['size_b'], row['chameleon_index']),
                    textcoords='offset points', xytext=(8, 6), fontsize=10,
                    alpha=0.85, ha='left')
    
    # Trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_plot['size_b'], df_plot['chameleon_index']
    )
    x_line = np.linspace(df_plot['size_b'].min(), df_plot['size_b'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color='#e74c3c', linewidth=2, alpha=0.7,
            label=f'Trend line (R\u00b2 = {r_value**2:.3f}, p = {p_value:.3f})')
    
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=17, fontweight='bold')
    ax.set_ylabel('Chameleon Index (CI)', fontsize=17, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14, loc='best')
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_xscale('log')
    
    plt.tight_layout()
    save_fig(fig, "figure6_size_vs_chameleon", cfg)
    plt.close()


def plot_figure6_1_size_vs_ipi_nocontext(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 6.1 (Supplementary): Scatter plot of Model Size (B params) vs IPI (No-Context).
    """
    from scipy import stats
    
    params_db = {
        'google/gemma-3-4b-it': 4,
        'google/gemma-3-12b-it': 12,
        'google/gemma-3-27b-it': 27,
        'mistralai/mixtral-8x7b-instruct-v0.1': 47,  # MoE total
        'mistralai/mistral-small-3.2-24b-instruct-2506': 24,
        'openai/gpt-oss-120b': 120,
        'openai/gpt-oss-20b': 20,
        'qwen/qwen3-14b': 14,
        'qwen/qwen3-32b': 32,
        'qwen/qwen3-235b-a22b-instruct-2507': 235,  # MoE total
        'meta-llama/meta-llama-3.1-8b-instruct': 8,
        'meta-llama/meta-llama-3.1-70b-instruct': 70,
        'meta-llama/llama-4-scout-17b-16e-instruct': 109,  # MoE total
        'deepseek-ai/deepseek-v3.2': 671,  # MoE total
        'microsoft/phi-4': 14,
        'nvidia/nvidia-nemotron-nano-12b-v2-vl': 12,
    }
    
    # Filter neutral only and get mean IPI per model
    df_neutral = df_ip[df_ip['tendencia'] == 'neutro'].copy()
    df_mean = df_neutral.groupby('modelo')['indice_polarizacao'].mean().reset_index()
    df_mean.columns = ['modelo', 'ipi_neutral']
    
    # Map model sizes
    df_mean['size_b'] = df_mean['modelo'].apply(lambda x: get_model_size(x, params_db))
    df_plot = df_mean.dropna(subset=['size_b']).copy()
    
    if df_plot.empty:
        print("Warning: No models with known sizes for scatter plot 6.1.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(df_plot['size_b'], df_plot['ipi_neutral'],
               s=120, color='#95a5a6', alpha=0.8, edgecolors='white', linewidth=1.2, zorder=3)
    
    for _, row in df_plot.iterrows():
        short_name = row['modelo'].split('/')[-1] if '/' in row['modelo'] else row['modelo']
        ax.annotate(short_name, (row['size_b'], row['ipi_neutral']),
                    textcoords='offset points', xytext=(8, 6), fontsize=10,
                    alpha=0.85, ha='left')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_plot['size_b'], df_plot['ipi_neutral']
    )
    x_line = np.linspace(df_plot['size_b'].min(), df_plot['size_b'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color='#e74c3c', linewidth=2, alpha=0.7,
            label=f'Trend line (R\u00b2 = {r_value**2:.3f}, p = {p_value:.3f})')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.4)
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=17, fontweight='bold')
    ax.set_ylabel('IPI (No-Context User)', fontsize=17, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14, loc='best')
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_xscale('log')
    
    plt.tight_layout()
    save_fig(fig, "figure6_1_size_vs_ipi_nocontext", cfg)
    plt.close()


def plot_figure7_agreement_heatmap(df_validos: pd.DataFrame, cfg: DictConfig):
    """
    Figure 7: Disaggregated Agreement Heatmap
    Rows: User tendency. Columns: Models.
    Cell value: Percentage of 'Agree' + 'Strongly Agree' responses.
    """
    # Filter only Agree + Strongly Agree (scores 1 and 2)
    df_validos_copy = df_validos.copy()
    df_validos_copy['is_agree'] = df_validos_copy['pontuacao'].isin([1, 2]).astype(int)
    
    # Calculate agreement percentage per model and tendency
    df_agree = df_validos_copy.groupby(['modelo', 'tendencia']).agg(
        total=('is_agree', 'count'),
        agree_count=('is_agree', 'sum')
    ).reset_index()
    df_agree['agree_pct'] = (df_agree['agree_count'] / df_agree['total']) * 100
    
    # Pivot for heatmap (transposed: tendencia as rows, modelo as columns)
    heatmap_data = df_agree.pivot(index='tendencia', columns='modelo', values='agree_pct')
    
    # Rename rows to English
    row_rename = {'neutro': 'No-Context', 'esquerda': 'Left-Wing', 'direita': 'Right-Wing'}
    heatmap_data.index = [row_rename.get(r, r) for r in heatmap_data.index]
    
    # Sort models by average agreement rate
    model_avg = heatmap_data.mean(axis=0).sort_values(ascending=True)
    heatmap_data = heatmap_data[model_avg.index]
    
    # Reorder rows
    row_order = ['Left-Wing', 'No-Context', 'Right-Wing']
    heatmap_data = heatmap_data.reindex([r for r in row_order if r in heatmap_data.index])
    
    fig, ax = plt.subplots(figsize=(24, 9))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Agreement Rate (%)'},
                linewidths=0.5, linecolor='white', vmin=0, vmax=100)
    
    cbar = ax.collections[0].colorbar
    cbar.set_label('Agreement Rate (%)', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=15)
    
    ax.set_xlabel('Model', fontsize=19, fontweight='bold')
    ax.set_ylabel('User Profile', fontsize=19, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=17)
    for text in ax.texts:
        text.set_fontsize(15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    save_fig(fig, "figure7_agreement_heatmap", cfg)
    plt.close()


def plot_figure8_swing_asymmetry(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    Figure 8: Swing Asymmetry (Grouped Bar Chart)
    Y-axis: Model names. X-axis: |ΔIPI|.
    Two bars per model: blue for |ΔIPI_left|, red for |ΔIPI_right|.
    """
    # Calculate mean IP per model and tendency
    df_shifts = df_ip.groupby(['modelo', 'tendencia'])['indice_polarizacao'].mean().reset_index()
    df_pivot = df_shifts.pivot(index='modelo', columns='tendencia', values='indice_polarizacao').reset_index()
    
    df_pivot['delta_left'] = abs(df_pivot['esquerda'] - df_pivot['neutro'])
    df_pivot['delta_right'] = abs(df_pivot['direita'] - df_pivot['neutro'])
    
    # Sort by total shift
    df_pivot['total_shift'] = df_pivot['delta_left'] + df_pivot['delta_right']
    df_pivot = df_pivot.sort_values('total_shift', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    y_pos = np.arange(len(df_pivot))
    bar_height = 0.35
    
    ax.barh(y_pos - bar_height / 2, df_pivot['delta_left'], bar_height,
            color='#e74c3c', alpha=0.85, label='|ΔIPI left|', edgecolor='white', linewidth=0.5)
    ax.barh(y_pos + bar_height / 2, df_pivot['delta_right'], bar_height,
            color='#3498db', alpha=0.85, label='|ΔIPI right|', edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_pivot['modelo'], fontsize=17)
    ax.set_xlabel('Magnitude of Shift |ΔIPI|', fontsize=19, fontweight='bold')
    ax.set_ylabel('Model', fontsize=19, fontweight='bold')
    ax.tick_params(axis='x', labelsize=15)
    ax.legend(fontsize=17, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    save_fig(fig, "figure8_swing_asymmetry", cfg)
    plt.close()


def plot_figure3_1_likert_comparison_by_tendency(
    df_orig: pd.DataFrame, df_neg: pd.DataFrame, cfg: DictConfig
):
    """
    Figure 3.1: Side-by-side Likert distributions – Original vs Negated statements,
    faceted by user tendency (Left / No-Context / Right).
    """
    likert_labels = ['Strongly\nDisagree', 'Disagree', 'Neutral', 'Agree', 'Strongly\nAgree']
    likert_values = [-2, -1, 0, 1, 2]
    tendencies = ['esquerda', 'neutro', 'direita']
    tend_titles = {'esquerda': 'Left-Wing User', 'neutro': 'No-Context User', 'direita': 'Right-Wing User'}

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

    for ax, tend in zip(axes, tendencies):
        # Original counts
        counts_orig = (
            df_orig[df_orig['tendencia'] == tend]
            .groupby('pontuacao').size()
            .reindex(likert_values, fill_value=0)
        )
        # Negated counts
        counts_neg = (
            df_neg[df_neg['tendencia'] == tend]
            .groupby('pontuacao').size()
            .reindex(likert_values, fill_value=0)
        )

        x = np.arange(len(likert_values))
        w = 0.35

        bars_o = ax.bar(x - w / 2, counts_orig.values, w,
                        label='Original', color='#2ecc71', alpha=0.85,
                        edgecolor='white', linewidth=0.5)
        bars_n = ax.bar(x + w / 2, counts_neg.values, w,
                        label='Negated', color='#9b59b6', alpha=0.85,
                        edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(likert_labels, fontsize=14)
        ax.set_title(tend_titles[tend], fontsize=20, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.tick_params(axis='y', labelsize=13)

    axes[0].set_ylabel('Response Count', fontsize=17, fontweight='bold')
    axes[1].set_xlabel('Likert Scale Response', fontsize=17, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
               fontsize=17, ncol=2, frameon=False)

    plt.tight_layout()
    save_fig(fig, "figure3_1_likert_orig_vs_negated", cfg)
    plt.close()


def plot_figure3_2_likert_proportion_shift(
    df_orig: pd.DataFrame, df_neg: pd.DataFrame, cfg: DictConfig
):
    """
    Figure 3.2: Proportional shift heatmap – for each model × tendency,
    shows the change in share of each Likert response when switching
    from Original to Negated statements (Negated% − Original%).
    """
    likert_values = [-2, -1, 0, 1, 2]
    likert_labels = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
    tendencies = ['esquerda', 'neutro', 'direita']
    tend_titles = {'esquerda': 'Left-Wing User', 'neutro': 'No-Context User', 'direita': 'Right-Wing User'}

    fig, axes = plt.subplots(1, 3, figsize=(28, 10), sharey=True)

    for ax, tend in zip(axes, tendencies):
        df_o_t = df_orig[df_orig['tendencia'] == tend]
        df_n_t = df_neg[df_neg['tendencia'] == tend]

        models = sorted(df_orig['modelo'].unique())

        shift_matrix = []
        for model in models:
            o_counts = (
                df_o_t[df_o_t['modelo'] == model]
                .groupby('pontuacao').size()
                .reindex(likert_values, fill_value=0)
            )
            n_counts = (
                df_n_t[df_n_t['modelo'] == model]
                .groupby('pontuacao').size()
                .reindex(likert_values, fill_value=0)
            )
            o_pct = o_counts / o_counts.sum() * 100 if o_counts.sum() > 0 else o_counts * 0
            n_pct = n_counts / n_counts.sum() * 100 if n_counts.sum() > 0 else n_counts * 0
            shift_matrix.append((n_pct - o_pct).values)

        shift_df = pd.DataFrame(shift_matrix, index=models, columns=likert_labels)

        vmax = max(abs(shift_df.values.min()), abs(shift_df.values.max()), 1)
        sns.heatmap(
            shift_df, ax=ax, cmap='RdBu_r', center=0,
            vmin=-vmax, vmax=vmax,
            annot=True, fmt='.1f', annot_kws={'size': 11},
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Δ Proportion (pp)', 'shrink': 0.8}
        )
        ax.set_title(tend_titles[tend], fontsize=20, fontweight='bold')
        ax.set_xlabel('', fontsize=1)
        ax.tick_params(axis='x', labelsize=12, rotation=35)
        ax.tick_params(axis='y', labelsize=13)

    axes[0].set_ylabel('Model', fontsize=17, fontweight='bold')
    fig.suptitle('Proportional Shift: Negated − Original (percentage points)',
                 fontsize=22, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_fig(fig, "figure3_2_likert_proportion_shift", cfg)
    plt.close()


def plot_figure3_3_likert_comparison_aggregated(
    df_orig: pd.DataFrame, df_neg: pd.DataFrame, cfg: DictConfig
):
    """
    Figure 3.3: Aggregated Likert distribution (all tendencies combined)
    comparing Original vs Negated statements.
    """
    likert_labels = ['Strongly\nDisagree', 'Disagree', 'Neutral', 'Agree', 'Strongly\nAgree']
    likert_values = [-2, -1, 0, 1, 2]

    counts_orig = (
        df_orig.groupby('pontuacao').size()
        .reindex(likert_values, fill_value=0)
    )
    counts_neg = (
        df_neg.groupby('pontuacao').size()
        .reindex(likert_values, fill_value=0)
    )

    x = np.arange(len(likert_values))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - w / 2, counts_orig.values, w,
           label='Original', color='#2ecc71', alpha=0.85,
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, counts_neg.values, w,
           label='Negated', color='#9b59b6', alpha=0.85,
           edgecolor='white', linewidth=0.5)

    # Add count labels on bars
    for i, (vo, vn) in enumerate(zip(counts_orig.values, counts_neg.values)):
        ax.text(i - w / 2, vo + max(counts_orig.max(), counts_neg.max()) * 0.01,
                str(int(vo)), ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(i + w / 2, vn + max(counts_orig.max(), counts_neg.max()) * 0.01,
                str(int(vn)), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(likert_labels, fontsize=16)
    ax.set_xlabel('Likert Scale Response', fontsize=17, fontweight='bold')
    ax.set_ylabel('Response Count', fontsize=17, fontweight='bold')
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=17, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    save_fig(fig, "figure3_3_likert_aggregated", cfg)
    plt.close()


def plot_figure9_judge_vs_nonjudge(df_ip: pd.DataFrame, df_validos: pd.DataFrame, cfg: DictConfig):
    """
    Figure 9: Judge-Model Neutrality Validation
    Two-panel figure proving that the 3 judge models (gpt-oss-120b,
    DeepSeek-V3.2, gemma-3-27b-it) do not behave anomalously compared
    to the 18 non-judge models.
    Panel A: Chameleon Index (CI)
    Panel B: Agreement Rate (%)
    Includes independent-samples t-test (or Mann-Whitney U) results.
    """
    JUDGE_MODELS = [
        'openai/gpt-oss-120b',
        'deepseek-ai/DeepSeek-V3.2',
        'google/gemma-3-27b-it',
    ]

    # ── Panel A: Chameleon Index ──────────────────────────────────────
    df_shifts = df_ip.groupby(['modelo', 'tendencia'])['indice_polarizacao'].mean().reset_index()
    df_pivot = df_shifts.pivot(index='modelo', columns='tendencia', values='indice_polarizacao').reset_index()
    df_pivot['shift_left'] = abs(df_pivot['esquerda'] - df_pivot['neutro'])
    df_pivot['shift_right'] = abs(df_pivot['direita'] - df_pivot['neutro'])
    df_pivot['chameleon_index'] = df_pivot['shift_left'] + df_pivot['shift_right']

    df_pivot['group'] = df_pivot['modelo'].apply(
        lambda m: 'Judge Models' if m in JUDGE_MODELS else 'Non-Judge Models'
    )

    ci_judge = df_pivot.loc[df_pivot['group'] == 'Judge Models', 'chameleon_index'].values
    ci_nonjudge = df_pivot.loc[df_pivot['group'] == 'Non-Judge Models', 'chameleon_index'].values

    # Statistical test for CI
    if len(ci_judge) >= 2 and len(ci_nonjudge) >= 2:
        _, p_shapiro_j = stats.shapiro(ci_judge) if len(ci_judge) >= 3 else (0, 1.0)
        _, p_shapiro_nj = stats.shapiro(ci_nonjudge)
        if p_shapiro_j > 0.05 and p_shapiro_nj > 0.05:
            stat_ci, p_ci = stats.ttest_ind(ci_judge, ci_nonjudge, equal_var=False)
            test_name_ci = "Welch's t-test"
        else:
            stat_ci, p_ci = stats.mannwhitneyu(ci_judge, ci_nonjudge, alternative='two-sided')
            test_name_ci = 'Mann-Whitney U'
    else:
        stat_ci, p_ci, test_name_ci = np.nan, np.nan, 'N/A'

    # ── Panel B: Agreement Rate ───────────────────────────────────────
    df_v = df_validos.copy()
    df_v['is_agree'] = df_v['pontuacao'].isin([1, 2]).astype(int)
    df_agree = df_v.groupby('modelo').agg(
        total=('is_agree', 'count'),
        agree_count=('is_agree', 'sum'),
    ).reset_index()
    df_agree['agree_pct'] = (df_agree['agree_count'] / df_agree['total']) * 100
    df_agree['group'] = df_agree['modelo'].apply(
        lambda m: 'Judge Models' if m in JUDGE_MODELS else 'Non-Judge Models'
    )

    ar_judge = df_agree.loc[df_agree['group'] == 'Judge Models', 'agree_pct'].values
    ar_nonjudge = df_agree.loc[df_agree['group'] == 'Non-Judge Models', 'agree_pct'].values

    # Statistical test for Agreement Rate
    if len(ar_judge) >= 2 and len(ar_nonjudge) >= 2:
        _, p_shapiro_j_ar = stats.shapiro(ar_judge) if len(ar_judge) >= 3 else (0, 1.0)
        _, p_shapiro_nj_ar = stats.shapiro(ar_nonjudge)
        if p_shapiro_j_ar > 0.05 and p_shapiro_nj_ar > 0.05:
            stat_ar, p_ar = stats.ttest_ind(ar_judge, ar_nonjudge, equal_var=False)
            test_name_ar = "Welch's t-test"
        else:
            stat_ar, p_ar = stats.mannwhitneyu(ar_judge, ar_nonjudge, alternative='two-sided')
            test_name_ar = 'Mann-Whitney U'
    else:
        stat_ar, p_ar, test_name_ar = np.nan, np.nan, 'N/A'

    # ── Plot ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    palette = {'Judge Models': '#e74c3c', 'Non-Judge Models': '#3498db'}
    group_order = ['Judge Models', 'Non-Judge Models']

    # Panel A – Chameleon Index
    sns.boxplot(
        data=df_pivot, x='group', y='chameleon_index', order=group_order,
        palette=palette, width=0.5, linewidth=1.5, ax=ax1,
        boxprops=dict(alpha=0.7), fliersize=0,
    )
    sns.stripplot(
        data=df_pivot, x='group', y='chameleon_index', order=group_order,
        palette=palette, size=10, alpha=0.85, jitter=0.15, ax=ax1,
        edgecolor='white', linewidth=0.8,
    )
    # Annotate judge model names
    judge_rows = df_pivot[df_pivot['group'] == 'Judge Models']
    for _, row in judge_rows.iterrows():
        short_name = row['modelo'].split('/')[-1]
        ax1.annotate(
            short_name,
            xy=(0, row['chameleon_index']),
            xytext=(15, 0), textcoords='offset points',
            fontsize=9, fontstyle='italic', color='#c0392b',
            arrowprops=dict(arrowstyle='-', color='#c0392b', lw=0.8),
        )

    ax1.set_xlabel('')
    ax1.set_ylabel('Chameleon Index (CI)', fontsize=17, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Significance annotation for CI
    p_label_ci = f'p = {p_ci:.3f}' if not np.isnan(p_ci) else 'N/A'
    sig_ci = 'n.s.' if (not np.isnan(p_ci) and p_ci > 0.05) else ('*' if not np.isnan(p_ci) else '')
    ax1.set_title(f'A   {test_name_ci}: {p_label_ci} ({sig_ci})', fontsize=15, fontweight='bold', loc='left')

    # Panel B – Agreement Rate
    sns.boxplot(
        data=df_agree, x='group', y='agree_pct', order=group_order,
        palette=palette, width=0.5, linewidth=1.5, ax=ax2,
        boxprops=dict(alpha=0.7), fliersize=0,
    )
    sns.stripplot(
        data=df_agree, x='group', y='agree_pct', order=group_order,
        palette=palette, size=10, alpha=0.85, jitter=0.15, ax=ax2,
        edgecolor='white', linewidth=0.8,
    )
    # Annotate judge model names
    judge_agree = df_agree[df_agree['group'] == 'Judge Models']
    for _, row in judge_agree.iterrows():
        short_name = row['modelo'].split('/')[-1]
        ax2.annotate(
            short_name,
            xy=(0, row['agree_pct']),
            xytext=(15, 0), textcoords='offset points',
            fontsize=9, fontstyle='italic', color='#c0392b',
            arrowprops=dict(arrowstyle='-', color='#c0392b', lw=0.8),
        )

    ax2.set_xlabel('')
    ax2.set_ylabel('Agreement Rate (%)', fontsize=17, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    p_label_ar = f'p = {p_ar:.3f}' if not np.isnan(p_ar) else 'N/A'
    sig_ar = 'n.s.' if (not np.isnan(p_ar) and p_ar > 0.05) else ('*' if not np.isnan(p_ar) else '')
    ax2.set_title(f'B   {test_name_ar}: {p_label_ar} ({sig_ar})', fontsize=15, fontweight='bold', loc='left')

    # Summary stats printed to console
    print("\n══════════ Figure 9 – Judge-Model Neutrality Validation ══════════")
    print(f"  Chameleon Index  → Judge mean={ci_judge.mean():.3f}  Non-Judge mean={ci_nonjudge.mean():.3f}")
    print(f"                     {test_name_ci}: stat={stat_ci:.4f}, p={p_ci:.4f}")
    print(f"  Agreement Rate   → Judge mean={ar_judge.mean():.1f}%  Non-Judge mean={ar_nonjudge.mean():.1f}%")
    print(f"                     {test_name_ar}: stat={stat_ar:.4f}, p={p_ar:.4f}")
    print("══════════════════════════════════════════════════════════════════\n")

    plt.tight_layout()
    save_fig(fig, "figure9_judge_neutrality_validation", cfg)
    plt.close()


def plot_figure10_pipeline_role_neutrality(df_ip: pd.DataFrame, df_validos: pd.DataFrame, cfg: DictConfig):
    """
    Figure 10: Pipeline-Role Neutrality Validation (3 groups)
    Proves that models with special pipeline roles (Judges and Pair Generators)
    do not behave anomalously compared to the 16 Pure Respondents.
    Panel A: Chameleon Index (CI)
    Panel B: Agreement Rate (%)
    Uses Kruskal-Wallis H-test (non-parametric, 3 groups) with pairwise
    Mann-Whitney U post-hoc.
    """
    JUDGE_MODELS = [
        'openai/gpt-oss-120b',
        'deepseek-ai/DeepSeek-V3.2',
        'google/gemma-3-27b-it',
    ]
    PAIR_GEN_MODELS = [
        'google/gemini-2.5-flash',
        'grok-4-1-fast-reasoning',
    ]

    def _assign_role(model: str) -> str:
        if model in JUDGE_MODELS:
            return 'Judge Models'
        if model in PAIR_GEN_MODELS:
            return 'Pair Generators'
        return 'Pure Respondents'

    ROLE_ORDER = ['Judge Models', 'Pair Generators', 'Pure Respondents']
    ROLE_PALETTE = {
        'Judge Models': '#e74c3c',
        'Pair Generators': '#f39c12',
        'Pure Respondents': '#3498db',
    }

    # ── Chameleon Index per model ─────────────────────────────────────
    df_shifts = df_ip.groupby(['modelo', 'tendencia'])['indice_polarizacao'].mean().reset_index()
    df_pivot = df_shifts.pivot(index='modelo', columns='tendencia', values='indice_polarizacao').reset_index()
    df_pivot['shift_left'] = abs(df_pivot['esquerda'] - df_pivot['neutro'])
    df_pivot['shift_right'] = abs(df_pivot['direita'] - df_pivot['neutro'])
    df_pivot['chameleon_index'] = df_pivot['shift_left'] + df_pivot['shift_right']
    df_pivot['role'] = df_pivot['modelo'].apply(_assign_role)

    # ── Agreement Rate per model ──────────────────────────────────────
    df_v = df_validos.copy()
    df_v['is_agree'] = df_v['pontuacao'].isin([1, 2]).astype(int)
    df_agree = df_v.groupby('modelo').agg(
        total=('is_agree', 'count'),
        agree_count=('is_agree', 'sum'),
    ).reset_index()
    df_agree['agree_pct'] = (df_agree['agree_count'] / df_agree['total']) * 100
    df_agree['role'] = df_agree['modelo'].apply(_assign_role)

    # ── Kruskal-Wallis (omnibus, 3 groups) ────────────────────────────
    groups_ci = [df_pivot.loc[df_pivot['role'] == r, 'chameleon_index'].values for r in ROLE_ORDER]
    groups_ar = [df_agree.loc[df_agree['role'] == r, 'agree_pct'].values for r in ROLE_ORDER]

    try:
        kw_stat_ci, kw_p_ci = stats.kruskal(*groups_ci)
    except ValueError:
        kw_stat_ci, kw_p_ci = np.nan, np.nan
    try:
        kw_stat_ar, kw_p_ar = stats.kruskal(*groups_ar)
    except ValueError:
        kw_stat_ar, kw_p_ar = np.nan, np.nan

    # ── Pairwise Mann-Whitney U ───────────────────────────────────────
    pair_combos = [
        ('Judge Models', 'Pure Respondents'),
        ('Pair Generators', 'Pure Respondents'),
        ('Judge Models', 'Pair Generators'),
    ]
    pw_results = {}
    for metric_label, df_src, col in [
        ('CI', df_pivot, 'chameleon_index'),
        ('AR', df_agree, 'agree_pct'),
    ]:
        for r1, r2 in pair_combos:
            g1 = df_src.loc[df_src['role'] == r1, col].values
            g2 = df_src.loc[df_src['role'] == r2, col].values
            if len(g1) >= 2 and len(g2) >= 2:
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            else:
                u_stat, u_p = np.nan, np.nan
            pw_results[(metric_label, r1, r2)] = (u_stat, u_p)

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("Figure 10 – Pipeline-Role Neutrality (Judge / PairGen / Pure)")
    print("═" * 72)
    for r in ROLE_ORDER:
        ci_sub = df_pivot[df_pivot['role'] == r]
        ar_sub = df_agree[df_agree['role'] == r]
        print(f"\n  {r} (N={len(ci_sub)}):")
        names = ', '.join(ci_sub['modelo'].tolist())
        print(f"    Models : {names}")
        print(f"    CI  mean={ci_sub['chameleon_index'].mean():.3f}  std={ci_sub['chameleon_index'].std():.3f}")
        print(f"    AR  mean={ar_sub['agree_pct'].mean():.1f}%  std={ar_sub['agree_pct'].std():.1f}%")

    _ns = lambda p: '✓ n.s.' if (not np.isnan(p) and p > 0.05) else '✗ sig.'
    print(f"\n  Kruskal-Wallis (CI):  H={kw_stat_ci:.3f}, p={kw_p_ci:.4f}  {_ns(kw_p_ci)}")
    print(f"  Kruskal-Wallis (AR):  H={kw_stat_ar:.3f}, p={kw_p_ar:.4f}  {_ns(kw_p_ar)}")
    print("\n  Pairwise Mann-Whitney U:")
    for (ml, r1, r2), (u, p) in pw_results.items():
        sig = 'n.s.' if (np.isnan(p) or p > 0.05) else 'sig.'
        print(f"    {ml}: {r1} vs {r2}  U={u:.1f}  p={p:.4f}  {sig}")
    print("═" * 72 + "\n")

    # ── Figure ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    metrics = [
        (ax1, df_pivot, 'chameleon_index', 'Chameleon Index (CI)', kw_stat_ci, kw_p_ci),
        (ax2, df_agree, 'agree_pct', 'Agreement Rate (%)', kw_stat_ar, kw_p_ar),
    ]

    for ax, df_src, col, ylabel, kw_stat, kw_p in metrics:
        # Boxplot
        sns.boxplot(
            data=df_src, x='role', y=col, order=ROLE_ORDER,
            palette=ROLE_PALETTE, width=0.5, linewidth=1.5,
            boxprops=dict(alpha=0.35), fliersize=0, ax=ax,
        )
        # Individual points
        sns.stripplot(
            data=df_src, x='role', y=col, order=ROLE_ORDER,
            palette=ROLE_PALETTE, size=10, edgecolor='white',
            linewidth=0.8, jitter=0.12, ax=ax,
        )

        # Annotate special-role models only (Judge + PairGen)
        special = df_src[df_src['role'].isin(['Judge Models', 'Pair Generators'])]
        for _, row in special.iterrows():
            role_idx = ROLE_ORDER.index(row['role'])
            short = row['modelo'].split('/')[-1]
            ax.annotate(
                short,
                xy=(role_idx, row[col]),
                xytext=(12, 2), textcoords='offset points',
                fontsize=8.5, fontstyle='italic',
                color=ROLE_PALETTE[row['role']],
                arrowprops=dict(arrowstyle='-', color=ROLE_PALETTE[row['role']], lw=0.7),
            )

        # Group-mean dashes
        for i, role in enumerate(ROLE_ORDER):
            mean_val = df_src.loc[df_src['role'] == role, col].mean()
            ax.hlines(mean_val, i - 0.18, i + 0.18,
                      colors='black', linewidth=2, linestyle='--', zorder=5)

        # Kruskal-Wallis annotation box
        sig_label = 'n.s.' if (not np.isnan(kw_p) and kw_p > 0.05) else 'p < 0.05'
        ax.text(
            0.98, 0.97,
            f'Kruskal-Wallis H = {kw_stat:.2f}\np = {kw_p:.4f} ({sig_label})',
            transform=ax.transAxes, fontsize=10,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9),
        )

        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=17, fontweight='bold')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', alpha=0.3, linestyle=':')

    ax1.text(-0.12, 1.03, 'A', transform=ax1.transAxes,
             fontsize=22, fontweight='bold', va='top', ha='right')
    ax2.text(-0.08, 1.03, 'B', transform=ax2.transAxes,
             fontsize=22, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    save_fig(fig, "figure10_pipeline_role_neutrality", cfg)
    plt.close()


def plot_figure11_likert_violins_by_role(df_validos: pd.DataFrame, cfg: DictConfig):
    """
    Figure 11: Likert Response Distribution per Model (Violin Plot)
    Each response is mapped to 1–5 (Strongly Disagree=1 … Strongly Agree=5).
    Models are grouped by pipeline role (Pair Generators, Judge Models,
    Pure Respondents) with vertical separator lines.
    Pairwise Wilcoxon rank-sum tests are shown between groups.
    """
    # ── Role assignment ───────────────────────────────────────────────
    JUDGE_MODELS = [
        'openai/gpt-oss-120b',
        'deepseek-ai/DeepSeek-V3.2',
        'google/gemma-3-27b-it',
    ]
    PAIR_GEN_MODELS = [
        'google/gemini-2.5-flash',
        'grok-4-1-fast-reasoning',
    ]

    def _assign_role(model: str) -> str:
        if model in JUDGE_MODELS:
            return 'Judge Models'
        if model in PAIR_GEN_MODELS:
            return 'Pair Generators'
        return 'Pure Respondents'

    GROUP_ORDER = ['Pair Generators', 'Judge Models', 'Pure Respondents']
    GROUP_PALETTE = {
        'Pair Generators': '#f39c12',
        'Judge Models': '#e74c3c',
        'Pure Respondents': '#3498db',
    }

    # ── Map pontuacao (-2..2) → Likert 1..5 ──────────────────────────
    df = df_validos.copy()
    df['likert_score'] = df['pontuacao'] + 3   # -2→1, -1→2, 0→3, 1→4, 2→5
    df['role'] = df['modelo'].apply(_assign_role)

    # ── Build ordered model list (grouped) ────────────────────────────
    models_ordered = []
    group_boundaries = []   # x-positions of separator lines
    for grp in GROUP_ORDER:
        grp_models = sorted(df.loc[df['role'] == grp, 'modelo'].unique())
        models_ordered.extend(grp_models)
        group_boundaries.append(len(models_ordered))

    # Short labels for x-axis
    short_labels = [m.split('/')[-1] for m in models_ordered]

    # ── Create a categorical column to enforce ordering ───────────────
    df['modelo_cat'] = pd.Categorical(df['modelo'], categories=models_ordered, ordered=True)

    # ── Assign colour per model based on role ─────────────────────────
    model_colors = [GROUP_PALETTE[_assign_role(m)] for m in models_ordered]

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(26, 10))

    # Violin plot
    sns.violinplot(
        data=df, x='modelo_cat', y='likert_score',
        order=models_ordered, hue='modelo_cat', palette=model_colors,
        inner='box', cut=0, linewidth=1.2, ax=ax,
        density_norm='width', saturation=0.75, legend=False,
    )

    # Group-mean markers
    for i, m in enumerate(models_ordered):
        mean_val = df.loc[df['modelo'] == m, 'likert_score'].mean()
        ax.plot(i, mean_val, marker='D', color='black', markersize=6, zorder=5)

    # ── Separator lines between groups ────────────────────────────────
    for boundary in group_boundaries[:-1]:      # skip the last one (end)
        ax.axvline(x=boundary - 0.5, color='grey', linewidth=2,
                   linestyle='--', alpha=0.7, zorder=4)

    # ── Group labels at top ───────────────────────────────────────────
    prev = 0
    for grp, boundary in zip(GROUP_ORDER, group_boundaries):
        mid = (prev + boundary - 1) / 2.0
        ax.text(mid, 5.45, grp, ha='center', va='bottom',
                fontsize=14, fontweight='bold',
                color=GROUP_PALETTE[grp])
        prev = boundary

    # ── Omnibus tests: Kruskal-Wallis + Friedman + Kendall's W ───────
    groups_kw = [df.loc[df['role'] == r, 'likert_score'].values for r in GROUP_ORDER]
    try:
        kw_stat, kw_p = stats.kruskal(*groups_kw)
    except ValueError:
        kw_stat, kw_p = np.nan, np.nan

    df_friedman = (
        df.groupby(['pair_id', 'role'])['likert_score']
        .mean()
        .reset_index()
        .pivot(index='pair_id', columns='role', values='likert_score')
        .dropna()
    )
    df_friedman = df_friedman[GROUP_ORDER]

    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(
            df_friedman[GROUP_ORDER[0]].values,
            df_friedman[GROUP_ORDER[1]].values,
            df_friedman[GROUP_ORDER[2]].values,
        )
        kendall_w = friedman_stat / (len(df_friedman) * (len(GROUP_ORDER) - 1))
    except (ValueError, ZeroDivisionError):
        friedman_stat, friedman_p, kendall_w = np.nan, np.nan, np.nan

    stat_text = (
        f"Kruskal-Wallis H = {kw_stat:.2f}, p = {kw_p:.2e}\n"
        f"Friedman χ² = {friedman_stat:.2f}, p = {friedman_p:.2e}\n"
        f"Kendall's W = {kendall_w:.4f}"
    )
    ax.text(
        0.99, 0.98, stat_text,
        transform=ax.transAxes, fontsize=9,
        va='top', ha='right', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.92),
    )

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("Figure 11 – Likert Violin Plots by Pipeline Role")
    print("═" * 72)
    for grp in GROUP_ORDER:
        sub = df[df['role'] == grp]
        print(f"\n  {grp}  (N responses = {len(sub)}):")
        print(f"    Mean = {sub['likert_score'].mean():.3f}, "
              f"Median = {sub['likert_score'].median():.1f}, "
              f"Std = {sub['likert_score'].std():.3f}")
    print(f"\n  Friedman blocks (pair_id with all groups): {len(df_friedman)}")
    print(f"\n  {stat_text}")
    print("═" * 72 + "\n")

    # ── Axes formatting ───────────────────────────────────────────────
    ax.set_xlabel('Model', fontsize=17, fontweight='bold')
    ax.set_ylabel('Likert Response',
                  fontsize=17, fontweight='bold')
    ax.set_xticks(range(len(models_ordered)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=14)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1\nStrongly\nDisagree', '2\nDisagree',
                        '3\nNeutral', '4\nAgree',
                        '5\nStrongly\nAgree'], fontsize=14)
    ax.set_ylim(0.3, 5.9)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.axhline(y=3, color='grey', linewidth=1, linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_fig(fig, "figure11_likert_violins_by_role", cfg)
    plt.close()
