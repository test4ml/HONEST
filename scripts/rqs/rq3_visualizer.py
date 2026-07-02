#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3 Visualizer Module

This module provides visualization functions for RQ3 analysis results.
All visualizations follow a consistent style and are publication-ready.

Visualization Types:
1. Correlation plots (scatter + trend)
2. Group comparison bar charts
3. Distribution plots
4. Heatmaps

Usage:
    from rq3_visualizer import RQ3Visualizer

    visualizer = RQ3Visualizer(output_dir)
    visualizer.plot_correlations(features_df, correlation_results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from scipy.ndimage import gaussian_filter1d

# Import configuration
from rq3_config import Config, FEATURE_DEFINITIONS


# =============================================================================
# Visualizer Class
# =============================================================================

class RQ3Visualizer:
    """
    Visualization class for RQ3 analysis results.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir = self.output_dir / "figures" / "pdf"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib/seaborn style for single-column paper."""
        plt.style.use("default")
        sns.set_palette("husl")
        plt.rcParams.update({
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.dpi": Config.FIGURE_DPI,
            "savefig.dpi": Config.FIGURE_DPI,
            "savefig.bbox": "tight",
            "lines.linewidth": 0.8,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.3,
            "patch.linewidth": 0.4,
        })

    def save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save figure to output directory as both PNG and PDF."""
        png_path = self.figures_dir / f"{name}.png"
        pdf_path = self.pdf_dir / f"{name}.pdf"
        fig.savefig(png_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return png_path

    # -------------------------------------------------------------------------
    # Correlation Plots
    # -------------------------------------------------------------------------

    def plot_correlation_scatter(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str = "inconsistent",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Create scatter plot with trend line for continuous feature vs target.

        Args:
            df: DataFrame with feature and target columns
            feature: Feature column name
            target: Target column name (default: "inconsistent")
            title: Optional custom title
            filename: Optional custom filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_SINGLE)

        # For binary target, create jittered scatter
        x = df[feature].values
        y = df[target].values

        # Add small jitter to binary y values for visualization
        y_jitter = y + np.random.normal(0, 0.02, len(y))

        scatter = ax.scatter(
            x, y_jitter,
            c=y,
            cmap="RdYlGn_r",
            alpha=0.3,
            s=3,
            edgecolors="none",
        )

        # Add smoothed trend line (binned average)
        n_bins = min(20, df[feature].nunique())
        df_binned = df.copy()
        df_binned["bin"] = pd.cut(df[feature], bins=n_bins, labels=False)
        bin_stats = df_binned.groupby("bin").agg({
            feature: "mean",
            target: "mean"
        }).dropna()

        if len(bin_stats) > 1:
            ax.plot(
                bin_stats[feature],
                bin_stats[target] * 100,
                "r-",
                linewidth=2,
                label="Trend (binned avg)",
            )

        ax.set_xlabel(self._format_feature_name(feature))
        ax.set_ylabel("Inconsistency Rate (%)")
        ax.set_title(title or f"{self._format_feature_name(feature)} vs Inconsistency")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if filename is None:
            filename = f"scatter_{feature}"

        return self.save_figure(fig, filename)

    def plot_correlation_bars(
        self,
        correlation_results: List[Dict],
        title: str = "Feature Correlations with Inconsistency",
        filename: str = "correlation_bars",
    ) -> Path:
        """
        Create horizontal bar chart of correlation coefficients.

        Args:
            correlation_results: List of correlation result dicts
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        # Prepare data
        features = [r["feature"] for r in correlation_results]
        correlations = [r["pearson_r"] for r in correlation_results]
        p_values = [r["pearson_p"] for r in correlation_results]

        # Sort by absolute correlation
        sorted_idx = np.argsort(np.abs(correlations))[::-1]
        features = [features[i] for i in sorted_idx]
        correlations = [correlations[i] for i in sorted_idx]
        p_values = [p_values[i] for i in sorted_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=(3.5, max(2.6, len(features) * 0.28)))

        # Color by significance and direction
        colors = []
        for r, p in zip(correlations, p_values):
            if p < Config.ALPHA:
                colors.append("coral" if r > 0 else "steelblue")
            else:
                colors.append("gray")

        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, correlations, color=colors, edgecolor="black", alpha=0.7,
                       linewidth=0.4)

        # Add significance markers
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            if p < 0.001:
                marker = "***"
            elif p < 0.01:
                marker = "**"
            elif p < Config.ALPHA:
                marker = "*"
            else:
                marker = ""

            x_pos = bar.get_width() + 0.002 * np.sign(bar.get_width())
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, marker,
                    ha="left" if bar.get_width() > 0 else "right",
                    va="center", fontsize=6, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([self._format_feature_name(f) for f in features])
        ax.set_xlabel("Pearson Correlation Coefficient")
        ax.set_title(title)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="coral", edgecolor="black", label="Positive (sig.)"),
            Patch(facecolor="steelblue", edgecolor="black", label="Negative (sig.)"),
            Patch(facecolor="gray", edgecolor="black", label="Not significant"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        return self.save_figure(fig, filename)

    # -------------------------------------------------------------------------
    # Group Comparison Plots
    # -------------------------------------------------------------------------

    def plot_group_comparison(
        self,
        group_results: List[Dict],
        title: str,
        filename: str,
        show_ci: bool = True,
    ) -> Path:
        """
        Create bar chart comparing inconsistency rates across groups.

        Args:
            group_results: List of group comparison result dicts
            title: Plot title
            filename: Output filename
            show_ci: Whether to show confidence intervals

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_SINGLE)

        groups = [r["group_name"] for r in group_results]
        rates = [r["inconsistency_rate"] * 100 for r in group_results]
        n_samples = [r["n_samples"] for r in group_results]

        if show_ci:
            ci_lowers = [r["ci_lower"] * 100 for r in group_results]
            ci_uppers = [r["ci_upper"] * 100 for r in group_results]
            yerr = [
                [r - cl for r, cl in zip(rates, ci_lowers)],
                [cu - r for r, cu in zip(rates, ci_uppers)],
            ]
        else:
            yerr = None

        x_pos = np.arange(len(groups))
        bars = ax.bar(
            x_pos, rates,
            color=Config.COLORS["primary"],
            edgecolor="black",
            alpha=0.7,
            yerr=yerr,
            capsize=2,
            linewidth=0.4,
        )

        # Add value labels
        for bar, rate, n in zip(bars, rates, n_samples):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{rate:.1f}%\n(n={n:,})",
                ha="center",
                va="bottom",
                fontsize=5,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_ylabel("Inconsistency Rate (%)")
        ax.set_title(title, fontweight="bold", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(rates) * 1.4)

        plt.tight_layout()
        return self.save_figure(fig, filename)

    def plot_quantile_analysis(
        self,
        quantile_results: List[Dict],
        feature: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Create bar chart for quantile-based analysis.

        Args:
            quantile_results: List of quantile group results
            feature: Feature name for title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        title = f"Inconsistency Rate by {self._format_feature_name(feature)} Quartile"

        if filename is None:
            filename = f"quantile_{feature}"

        return self.plot_group_comparison(
            quantile_results,
            title=title,
            filename=filename,
        )

    # -------------------------------------------------------------------------
    # Distribution Plots
    # -------------------------------------------------------------------------

    def plot_feature_distribution(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str = "inconsistent",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Create distribution plot comparing consistent vs inconsistent samples.

        Args:
            df: DataFrame with feature and target columns
            feature: Feature to plot
            target: Target column
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIGURE_SIZE_DOUBLE)

        # Histogram
        consistent = df[df[target] == 0][feature]
        inconsistent = df[df[target] == 1][feature]

        ax1.hist(
            [consistent, inconsistent],
            bins=30,
            label=["Consistent", "Inconsistent"],
            color=[Config.COLORS["positive"], Config.COLORS["negative"]],
            alpha=0.6,
            edgecolor="black",
        )
        ax1.set_xlabel(self._format_feature_name(feature))
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Distribution of {self._format_feature_name(feature)}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        bp_data = [consistent, inconsistent]
        bp = ax2.boxplot(
            bp_data,
            labels=["Consistent", "Inconsistent"],
            patch_artist=True,
            showmeans=True,
        )

        colors = [Config.COLORS["positive"], Config.COLORS["negative"]]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel(self._format_feature_name(feature))
        ax2.set_title(f"{self._format_feature_name(feature)} by Consistency")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if filename is None:
            filename = f"distribution_{feature}"

        return self.save_figure(fig, filename)

    def plot_overall_distribution(
        self,
        df: pd.DataFrame,
        features: List[str],
        filename: str = "feature_distributions",
    ) -> Path:
        """
        Create grid of distribution plots for multiple features.

        Args:
            df: DataFrame with features
            features: List of features to plot
            filename: Output filename

        Returns:
            Path to saved figure
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5, 1.5 * n_rows),
            squeeze=False,
        )

        for idx, feature in enumerate(features):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            data = df[feature].dropna()
            ax.hist(data, bins=20, color=Config.COLORS["primary"],
                    edgecolor="black", alpha=0.7, linewidth=0.3)
            ax.axvline(data.mean(), color="red", linestyle="--",
                       linewidth=0.8, label=f'Mean: {data.mean():.2f}')
            ax.set_xlabel(self._format_feature_name(feature))
            ax.set_ylabel("Frequency")
            ax.set_title(self._format_feature_name(feature), fontsize=7)
            ax.legend(fontsize=5)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(features), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle("Feature Distributions", fontsize=8, fontweight="bold")
        plt.tight_layout()

        return self.save_figure(fig, filename)

    # -------------------------------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        features: List[str],
        title: str = "Feature Correlation Matrix",
        filename: str = "correlation_heatmap",
    ) -> Path:
        """
        Create correlation heatmap for features.

        Args:
            df: DataFrame with features
            features: List of features to include
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        # Calculate correlation matrix
        corr_matrix = df[features].corr()

        # Create figure
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_HEATMAP)

        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(features)))

        labels = [self._format_feature_name(f).replace(" ", "\n") for f in features]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)

        # Add correlation values
        for i in range(len(features)):
            for j in range(len(features)):
                val = corr_matrix.iloc[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=4.5)

        plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
        ax.set_title(title, fontweight="bold", fontsize=8)

        plt.tight_layout()
        return self.save_figure(fig, filename)

    # -------------------------------------------------------------------------
    # Summary Dashboard
    # -------------------------------------------------------------------------

    def create_summary_dashboard(
        self,
        df: pd.DataFrame,
        correlation_results: List[Dict],
        categorical_results: Dict[str, Tuple],
        filename: str = "summary_dashboard",
    ) -> Path:
        """
        Create comprehensive summary dashboard.

        Args:
            df: Features DataFrame
            correlation_results: Correlation analysis results
            categorical_results: Categorical analysis results
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=Config.FIGURE_SIZE_DASHBOARD)

        # 1. Overall inconsistency rate (top left)
        ax1 = fig.add_subplot(2, 3, 1)
        overall_rate = df["inconsistent"].mean() * 100
        ax1.bar(["Overall"], [overall_rate], color=Config.COLORS["primary"],
                edgecolor="black", alpha=0.7, linewidth=0.4)
        ax1.set_ylabel("Inconsistency Rate (%)")
        ax1.set_title("Overall Inconsistency Rate", fontweight="bold", fontsize=7)
        ax1.text(0, overall_rate + 1, f"{overall_rate:.1f}%", ha="center",
                 fontweight="bold", fontsize=6)
        ax1.set_ylim(0, overall_rate * 1.3)

        # 2. Top correlations (top middle)
        ax2 = fig.add_subplot(2, 3, 2)
        top_corrs = correlation_results[:5] if correlation_results else []
        if top_corrs:
            features = [r["feature"] for r in top_corrs]
            corrs = [r["pearson_r"] for r in top_corrs]
            colors = ["coral" if c > 0 else "steelblue" for c in corrs]
            y_pos = np.arange(len(features))
            ax2.barh(y_pos, corrs, color=colors, edgecolor="black", alpha=0.7,
                     linewidth=0.4)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([self._format_feature_name(f) for f in features],
                                fontsize=5)
            ax2.set_xlabel("Pearson r")
            ax2.set_title("Top 5 Correlations", fontweight="bold", fontsize=7)
            ax2.axvline(0, color="black", linewidth=0.5)

        # 3. Inconsistency by mutation type (top right)
        ax3 = fig.add_subplot(2, 3, 3)
        if "mutation_type" in categorical_results:
            group_results, _ = categorical_results["mutation_type"]
            groups = [r["group_name"] for r in group_results]
            rates = [r["inconsistency_rate"] * 100 for r in group_results]
            ax3.bar(groups, rates, color=Config.COLORS["primary"],
                    edgecolor="black", alpha=0.7, linewidth=0.4)
            ax3.set_ylabel("Inconsistency Rate (%)")
            ax3.set_title("By Mutation Type", fontweight="bold", fontsize=7)
            ax3.tick_params(axis="x", rotation=15, labelsize=5)

        # 4. Inconsistency by question type (middle left)
        ax4 = fig.add_subplot(2, 3, 4)
        if "question_type" in categorical_results:
            group_results, _ = categorical_results["question_type"]
            groups = [r["group_name"] for r in group_results]
            rates = [r["inconsistency_rate"] * 100 for r in group_results]
            ax4.bar(groups, rates, color=Config.COLORS["secondary"],
                    edgecolor="black", alpha=0.7, linewidth=0.4)
            ax4.set_ylabel("Inconsistency Rate (%)")
            ax4.set_title("By Question Type", fontweight="bold", fontsize=7)
            ax4.tick_params(axis="x", rotation=15, labelsize=5)

        # 5. Rule length distribution (middle)
        ax5 = fig.add_subplot(2, 3, 5)
        if "body_atom_count" in df.columns:
            consistent = df[df["inconsistent"] == 0]["body_atom_count"]
            inconsistent = df[df["inconsistent"] == 1]["body_atom_count"]
            ax5.hist([consistent, inconsistent], bins=range(1, 8),
                     label=["Consistent", "Inconsistent"],
                     color=[Config.COLORS["positive"], Config.COLORS["negative"]],
                     alpha=0.6, edgecolor="black", align="left", linewidth=0.3)
            ax5.set_xlabel("Body Atom Count")
            ax5.set_ylabel("Frequency")
            ax5.set_title("Rule Length Distribution", fontweight="bold", fontsize=7)
            ax5.legend(fontsize=5)

        # 6. Predicate scarcity effect (middle right)
        ax6 = fig.add_subplot(2, 3, 6)
        if "predicate_scarcity_score" in df.columns:
            df["scarcity_bin"] = pd.qcut(df["predicate_scarcity_score"],
                                          q=4, labels=["Q1", "Q2", "Q3", "Q4"],
                                          duplicates="drop")
            bin_stats = df.groupby("scarcity_bin", observed=False)["inconsistent"].mean() * 100
            ax6.bar(bin_stats.index, bin_stats.values,
                    color=Config.COLORS["primary"], edgecolor="black", alpha=0.7,
                    linewidth=0.4)
            ax6.set_xlabel("Predicate Scarcity Quartile")
            ax6.set_ylabel("Inconsistency Rate (%)")
            ax6.set_title("By Predicate Scarcity", fontweight="bold", fontsize=7)
            df.drop(columns=["scarcity_bin"], inplace=True)

        plt.suptitle("RQ3 Factor Analysis Summary Dashboard",
                     fontsize=9, fontweight="bold", y=1.02)
        plt.tight_layout()

        return self.save_figure(fig, filename)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _format_feature_name(self, feature: str) -> str:
        """Format feature name for display."""
        # Check if we have a definition
        if feature in FEATURE_DEFINITIONS:
            return FEATURE_DEFINITIONS[feature].get("description", feature)

        # Default formatting
        return feature.replace("_", " ").title()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Test the visualizer module."""

    print("=" * 70)
    print("RQ3 Visualizer Test")
    print("=" * 70)

    # Load data
    output_dir = Config.ensure_output_dir()
    features_path = output_dir / "features.csv"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        return

    print(f"\nLoading features from {features_path}...")
    features_df = pd.read_csv(features_path)

    # Create visualizer
    visualizer = RQ3Visualizer(output_dir)

    # Test individual plots
    print("\nGenerating test visualizations...")

    # Distribution plot
    if "body_atom_count" in features_df.columns:
        visualizer.plot_feature_distribution(features_df, "body_atom_count")
        print("  - Distribution plot created")

    # Scatter plot
    if "predicate_scarcity_score" in features_df.columns:
        visualizer.plot_correlation_scatter(features_df, "predicate_scarcity_score")
        print("  - Scatter plot created")

    print(f"\nVisualizations saved to: {visualizer.figures_dir}")

    print("\n" + "=" * 70)
    print("Visualizer test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
