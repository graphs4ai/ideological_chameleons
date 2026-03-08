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
    Rows: Models. Columns: User tendency.
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
    
    # Pivot for heatmap
    heatmap_data = df_agree.pivot(index='modelo', columns='tendencia', values='agree_pct')
    
    # Rename columns to English
    col_rename = {'neutro': 'No-Context', 'esquerda': 'Left-Wing', 'direita': 'Right-Wing'}
    heatmap_data.columns = [col_rename.get(c, c) for c in heatmap_data.columns]
    
    # Add average column and sort
    heatmap_data['Average'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values('Average', ascending=True)
    display_data = heatmap_data.drop('Average', axis=1)
    
    # Reorder columns
    col_order = ['Left-Wing', 'No-Context', 'Right-Wing']
    display_data = display_data[[c for c in col_order if c in display_data.columns]]
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    sns.heatmap(display_data, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Agreement Rate (%)'},
                linewidths=0.5, linecolor='white', vmin=0, vmax=100)
    
    cbar = ax.collections[0].colorbar
    cbar.set_label('Agreement Rate (%)', fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    
    ax.set_xlabel('User Profile', fontsize=17, fontweight='bold')
    ax.set_ylabel('Model', fontsize=17, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    for text in ax.texts:
        text.set_fontsize(14)
    plt.xticks(rotation=0)
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
