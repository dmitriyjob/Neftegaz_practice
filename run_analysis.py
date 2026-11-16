"""Reproduce the real BOEM case offline from its verified source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from cluster_wells import analyze, markdown_table, save_outputs
from prepare_boem import DEFAULT_DATA_DIR, prepare

ROOT = Path(__file__).resolve().parent
FEATURES = ["Oil_Rate_bpd", "Water_Cut_pct", "GOR_scf_bbl"]
LOG_FEATURES = ["Oil_Rate_bpd", "GOR_scf_bbl"]
COLORS = ["#087F8C", "#C77C24", "#4966A2", "#A64B66", "#6B8E48", "#7957A3", "#757575", "#A14F29"]


def json_write(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def transformed(frame: pd.DataFrame, features=FEATURES, logs=LOG_FEATURES, robust=False) -> np.ndarray:
    x = frame[list(features)].copy()
    for feature in logs:
        if feature in x:
            x[feature] = np.log1p(x[feature])
    return (RobustScaler() if robust else StandardScaler()).fit_transform(x)


def fit_labels(x: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)


def half_year_features(monthly: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    month = monthly.Production_Date.astype(str).str[-2:].astype(int)
    rows = monthly.loc[month.between(start, end)]
    sums = rows.groupby("API_Well_Number")[["Oil_bbl", "Gas_Mcf", "Water_bbl", "Days_On_Prod"]].sum()
    oil_months = rows.loc[rows.Oil_bbl.gt(0)].groupby("API_Well_Number").Production_Date.nunique()
    sums = sums.loc[(sums.Oil_bbl > 0) & (sums.Days_On_Prod > 0) & (oil_months >= 3)]
    return pd.DataFrame({
        "Oil_Rate_bpd": sums.Oil_bbl / sums.Days_On_Prod,
        "Water_Cut_pct": 100 * sums.Water_bbl / (sums.Oil_bbl + sums.Water_bbl),
        "GOR_scf_bbl": 1000 * sums.Gas_Mcf / sums.Oil_bbl,
    })


def sensitivity(wells: pd.DataFrame, result, monthly: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    k, labels = len(result.profiles), result.labels
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(result.scaled)
    checks = []
    for name, x, note in [
        ("RobustScaler", transformed(wells, robust=True), "Те же признаки и log1p; масштабирование по медиане и IQR, k фиксировано."),
        ("Без log1p", transformed(wells, logs=[]), "Те же признаки; StandardScaler, k фиксировано."),
        ("Без газового фактора", transformed(wells, features=FEATURES[:2]), "Дебит и обводнённость, log1p дебита; k фиксировано."),
    ]:
        checks.append({"name": name, "ari": float(adjusted_rand_score(labels, fit_labels(x, k))), "note": note})
    samples = []
    rng = np.random.default_rng(42)
    for i in range(10):
        indices = np.sort(rng.choice(len(wells), size=int(.8 * len(wells)), replace=False))
        refit = fit_labels(transformed(wells.iloc[indices]), k, 42 + i)
        samples.append({"repeat": i + 1, "n_wells": len(indices),
                        "ari": float(adjusted_rand_score(labels[indices], refit))})
    first, second = half_year_features(monthly, 1, 6), half_year_features(monthly, 7, 12)
    common = first.index.intersection(second.index).sort_values()
    h1 = fit_labels(transformed(first.loc[common]), k)
    h2 = fit_labels(transformed(second.loc[common]), k)
    checks.append({"name": "Первое и второе полугодия", "ari": float(adjusted_rand_score(h1, h2)),
                   "note": f"{len(common)} общих API с добычей нефти ≥3 месяцев в каждом полугодии; scaler и K-means обучены отдельно, k фиксировано."})
    return {
        "seed_ari_min": float(result.stability.adjusted_rand_index.min()),
        "seed_ari_max": float(result.stability.adjusted_rand_index.max()),
        "ward_ari": float(adjusted_rand_score(labels, ward)),
        "ward_silhouette": float(silhouette_score(result.scaled, ward)),
        "subsample_ari_min": min(r["ari"] for r in samples),
        "subsample_ari_median": float(np.median([r["ari"] for r in samples])),
        "subsample_ari_max": max(r["ari"] for r in samples),
        "checks": checks,
    }, pd.DataFrame(samples)


def make_plots(output: Path, wells: pd.DataFrame, result, xy: np.ndarray, variance: np.ndarray, sil: np.ndarray) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "#FAF9F5"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in result.profiles.index:
        mask = result.labels == c
        color = COLORS[c - 1]
        axes[0].scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=.55, color=color, label=f"Группа {c}: {mask.sum()}")
        axes[1].scatter(wells.loc[mask, "Oil_Rate_bpd"], wells.loc[mask, "Water_Cut_pct"], s=14, alpha=.55, color=color)
    axes[0].set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})", title="Проекция трёх признаков")
    axes[0].legend(fontsize=10)
    axes[1].set(xlabel="Нефть на день работы, барр./сут (log-шкала)", ylabel="Обводнённость, %", title="Профили в физических единицах", xscale="log")
    fig.tight_layout()
    fig.savefig(output / "well_groups.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    start = 0
    for c in result.profiles.index:
        values = np.sort(sil[result.labels == c])
        y = np.arange(start, start + len(values))
        ax.fill_betweenx(y, 0, values, color=COLORS[c - 1], alpha=.85)
        ax.text(.02, start + len(values) / 2, f"Группа {c}, n={len(values)}", fontsize=11)
        start += len(values) + 25
    ax.axvline(sil.mean(), linestyle="--", color="#222222", label=f"Среднее {sil.mean():.3f}")
    ax.set(xlabel="Индивидуальный силуэт", ylabel="Скважины, отсортированные внутри групп", yticks=[], title="Пограничные скважины видны по силуэту")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "silhouette_distribution.png", dpi=180)
    plt.close(fig)


def report(summary: dict) -> str:
    s = summary
    profiles = pd.DataFrame(s["clusters"])[["cluster", "n_wells", "oil_rate_bpd", "water_cut_pct", "gor_scf_bbl"]]
    lines = ["# Кластеризация нефтедобывающих скважин по данным BOEM", "",
        "Предмет: Анализ данных в нефтегазе 26/27. Тема 9. Основной воспроизводимый кейс.", "",
        f"Источник: [{s['source']['name']}]({s['source']['url']}). Период: январь–декабрь 2023 года. "
        f"Из {s['counts']['raw_rows']:,} completion-month записей по {s['counts']['raw_wells']} API "
        f"отобрано {s['counts']['selected_wells']} нефтяных стволов. Одна строка модели соответствует "
        "12-значному API, включая код бокового ствола. Это эксплуатационная сегментация выбранного фонда.", "",
        "## Подготовка и выборка", "", "[Паспорт источника и словарь](../../data/boem/README.md). "
        "[Все этапы отбора](../../data/boem/attrition.csv). Данные читаются из зафиксированного ZIP с проверкой SHA256.", "",
        markdown_table(pd.DataFrame(s["filters"])[["label", "before", "after"]]), "",
        "В выборке остаются API с нефтяным кодом, нефтедобывающими статусами активных записей, "
        "отчётностью за 12 месяцев, единственным активным completion и добычей нефти минимум в девяти месяцах. "
        "Это исключает двойной счёт дней и ограничивает выводы относительно регулярно работающим фондом. "
        "Новые, простаивающие и смешанные нефтегазовые объекты представлены неполно.", "",
        "## Признаки и модель", "",
        "Дебит нефти = годовая нефть / дни работы. Обводнённость = 100 × годовая вода / (нефть + вода). "
        "Газовый фактор = 1000 × газ в Mcf / нефть в баррелях. Используются отношения сумм; "
        "среднее месячных отношений отвечало бы другому вопросу. Пропусков в итоговых признаках нет.", "",
        "Для дебита и газового фактора применён log1p, затем все три признака стандартизованы. "
        "Так уменьшается влияние правых хвостов и различий единиц. Это выбор геометрии расстояния, "
        "его чувствительность проверена отдельно. Параметры K-means: n_init=20, random_state=42. "
        "Число групп выбирается по максимальному среднему силуэту среди k=2…8. "
        "Номера групп упорядочены по среднему дебиту; они не означают качество скважины.", "",
        f"Выбрано k={s['selected_k']}, силуэт {s['silhouette']:.3f}. "
        f"Доля объектов с отрицательным силуэтом {s['negative_silhouette_fraction']:.1%}. "
        "В отсутствие эталонных классов это характеристика геометрии, а не точность и не доказательство природных типов.", "",
        markdown_table(pd.DataFrame(s["k_scores"])), "", "![Выбор k](cluster_selection.png)", "",
        "## Профили и конкретные скважины", "",
        "В таблице ниже медианы индивидуальных годовых характеристик скважин в физических единицах. "
        "Медианы удобны при скошенных распределениях и не являются центрами K-means. "
        "Файл cluster_profiles.csv отдельно содержит средние тех же характеристик.", "", markdown_table(profiles), "",
        "Единицы: oil_rate_bpd = барр./сут работы; water_cut_pct = %; gor_scf_bbl = станд. куб. фут/барр.", "",
        "![Скважины](well_groups.png)", "",
        f"PCA показывает {sum(s['pca']['explained_variance_ratio']):.1%} дисперсии. Кластеризация выполнена "
        "по трём подготовленным признакам, проекция нужна только для изображения.", "",
        "Примеры выбраны как ближайшие к центрам групп в пространстве модели:", "",
        markdown_table(pd.DataFrame(s["examples"])[["well_id", "cluster", "oil_rate_bpd", "water_cut_pct", "gor_scf_bbl"]]), "",
        "Группа с большей обводнённостью помогает выбрать фонд для сравнения динамики воды и нагрузки на систему "
        "подготовки продукции. Для скважин с высоким газовым фактором полезны история давления, режимы сепарации "
        "и проверка учёта газа. По инструкции ONRR газлифтный газ уже вычитается из отчётной добычи. "
        "Более продуктивные скважины можно сравнивать с аналогами по "
        "сохранению добычи. Все эти направления являются задачами дополнительной проверки, а не назначением ремонта.", "",
        "## Устойчивость и ограничения", "",
        f"ARI при смене seed: {s['stability']['seed_ari_min']:.3f}…{s['stability']['seed_ari_max']:.3f}. "
        f"Для Ward при том же k: ARI {s['stability']['ward_ari']:.3f}, силуэт {s['stability']['ward_silhouette']:.3f}. "
        f"В десяти подвыборках по 80% объектов медиана ARI {s['stability']['subsample_ari_median']:.3f}, "
        f"диапазон {s['stability']['subsample_ari_min']:.3f}…{s['stability']['subsample_ari_max']:.3f}. "
        "Scaler и модель для подвыборок переобучаются; ARI сравнивается только на общих объектах.", "",
        markdown_table(pd.DataFrame(s["stability"]["checks"])), "",
        "При замене StandardScaler на RobustScaler совпадение разбиений заметно снижается. "
        "Поэтому устойчивость к seed и подвыборкам не означает независимость результата от "
        "подготовки признаков. Полученные группы полезны для выбранной схемы сравнения фонда, "
        "но не являются универсальными типами скважин.", "", "![Силуэты](silhouette_distribution.png)", ""]
    lines += ["- " + c for c in s["caveats"]]
    lines += ["", "## Воспроизведение", "", "```bash", s["reproduce_command"], "```", "",
              "Версии пакетов, хеши данных и исходников: [run_manifest.json](run_manifest.json). "
              "Теория, источники и итоговая интерпретация: [реферат](../../docs/report.md).", ""]
    return "\n".join(lines)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    provenance = prepare()
    id_columns = ["Well_ID", "API_Well_Number", "Lease_Number", "Completion_Name", "Production_Interval_Code", "Area_Block", "Field_Name_Code"]
    wells = pd.read_csv(DEFAULT_DATA_DIR / "well_features.csv", dtype={column: str for column in id_columns}).sort_values("Well_ID").set_index("Well_ID")
    monthly = pd.read_csv(DEFAULT_DATA_DIR / "monthly_selected.csv", dtype={"API_Well_Number": str, "Production_Date": str})
    result = analyze(wells.reset_index(), FEATURES, max_k=8, seed=42, log_features=LOG_FEATURES)
    save_outputs(result, output, "BOEM OGOR-A 2023, подготовленные годовые признаки")
    pca = PCA(n_components=2).fit(result.scaled)
    xy = pca.transform(result.scaled)
    sil = silhouette_samples(result.scaled, result.labels)
    wells["cluster"] = result.labels
    wells["silhouette"] = sil
    wells.to_csv(output / "well_clusters.csv", float_format="%.10g")
    summary_profiles = []
    examples = []
    for c in result.profiles.index:
        mask = result.labels == c
        median = wells.loc[mask, FEATURES].median()
        summary_profiles.append({"cluster": int(c), "n_wells": int(mask.sum()),
            "oil_rate_bpd": float(median.Oil_Rate_bpd), "water_cut_pct": float(median.Water_Cut_pct),
            "gor_scf_bbl": float(median.GOR_scf_bbl), "label": f"Группа {c}"})
        indices = np.flatnonzero(mask)
        center = result.scaled[mask].mean(axis=0)
        representative = indices[np.argmin(np.linalg.norm(result.scaled[mask] - center, axis=1))]
        row = wells.iloc[representative]
        examples.append({"well_id": str(wells.index[representative]), "cluster": int(c),
            "oil_rate_bpd": float(row.Oil_Rate_bpd), "water_cut_pct": float(row.Water_Cut_pct),
            "gor_scf_bbl": float(row.GOR_scf_bbl), "representative_role": "nearest_centroid"})
    stability, subsamples = sensitivity(wells, result, monthly)
    subsamples.to_csv(output / "subsample_stability.csv", index=False)
    wells[FEATURES].corr(method="spearman").to_csv(output / "feature_correlations.csv")
    pd.DataFrame(summary_profiles).to_csv(output / "cluster_medians.csv", index=False)
    pd.DataFrame(examples).to_csv(output / "representative_wells.csv", index=False)
    attrition = pd.read_csv(DEFAULT_DATA_DIR / "attrition.csv")
    labels = {"00_source": "Все API", "01_positive_oil": "Есть добыча нефти", "02_oil_code_only": "Только нефтяной код O",
              "03_oil_status": "Нефтедобывающий статус", "04_full_year_reporting": "12 месяцев отчётности",
              "05_single_active_completion": "Один активный completion", "06_one_active_record_per_month": "Без пересечения дней",
              "07_at_least_nine_oil_months": "Нефть минимум 9 месяцев", "08_no_injection": "Без закачки"}
    filters = [{"label": labels[r.Stage], "before": int(r.Wells_Before), "after": int(r.Wells_After), "reason": r.Reason} for _, r in attrition.iterrows()]
    summary = {
        "source": {"name": "BOEM OGOR-A 2023", "url": provenance["landing_page"], "download_url": provenance["download_url"],
                   "period": "2023 год (январь–декабрь)", "accessed_at": provenance["accessed_on"], "license_or_terms": provenance["reuse_terms"]},
        "counts": {"raw_rows": provenance["source_rows"], "raw_wells": provenance["source_api_wellbores"],
                   "selected_wells": len(wells), "selected_rows": len(monthly)},
        "filters": filters, "profile_statistic": "median",
        "features": [
            {"key": "Oil_Rate_bpd", "label": "Дебит нефти", "unit": "барр./сут работы", "formula": "Σ нефть / Σ дни работы", "transform": "log1p + StandardScaler"},
            {"key": "Water_Cut_pct", "label": "Обводнённость", "unit": "%", "formula": "100 × Σ вода / (Σ нефть + Σ вода)", "transform": "StandardScaler"},
            {"key": "GOR_scf_bbl", "label": "Газовый фактор", "unit": "станд. куб. фут/барр.", "formula": "1000 × Σ газ (Mcf) / Σ нефть (bbl)", "transform": "log1p + StandardScaler"}],
        "selected_k": len(result.profiles), "silhouette": float(sil.mean()),
        "negative_silhouette_fraction": float((sil < 0).mean()),
        "k_scores": result.scores.to_dict("records"), "clusters": summary_profiles, "examples": examples,
        "pca": {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "points": [{"well_id": well, "cluster": int(c), "pc1": float(x), "pc2": float(y)} for well, c, (x, y) in zip(wells.index, result.labels, xy)]},
        "stability": stability,
        "caveats": [
            "Отбор ограничен нефтяными стволами с регулярной добычей в 2023 году; выводы не распространяются на весь фонд BOEM.",
            f"API включает боковой ствол. Скважины одного месторождения связаны общими условиями; {len(wells)} объектов нельзя автоматически считать независимой случайной выборкой.",
            "Дебит и газовый фактор имеют общий нефтяной объём в формулах. Корреляции и масштабирование влияют на геометрию групп.",
            "Годовые показатели скрывают динамику. Сравнение полугодий совмещает реальное изменение режима и чувствительность модели.",
            "В данных нет пластового давления, проницаемости, затрат и причин изменения режима. Кластеры не доказывают геологическую причинность или выгоду ремонта.",
            "Максимум силуэта среди k≥2 не является проверкой существования естественных кластеров. ARI не является точностью без эталонной разметки."],
        "reproduce_command": "python run_analysis.py",
    }
    json_write(output / "summary.json", summary)
    files = [ROOT / "run_analysis.py", ROOT / "prepare_boem.py", ROOT / "cluster_wells.py", ROOT / "requirements.txt", DEFAULT_DATA_DIR / "well_features.csv", DEFAULT_DATA_DIR / provenance["archive_name"]]
    json_write(output / "run_manifest.json", {
        "command": "python run_analysis.py", "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy"]},
        "features": FEATURES, "log_features": LOG_FEATURES, "scaler": "StandardScaler", "seed": 42, "n_init": 20, "k_range": [2, 8],
        "input_and_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    })
    make_plots(output, wells, result, xy, pca.explained_variance_ratio_, sil)
    report_content = report(summary).replace("../../data/", os.path.relpath(ROOT / "data", output) + "/").replace("../../docs/", os.path.relpath(ROOT / "docs", output) + "/")
    (output / "report.md").write_text(report_content, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "boem_2023")
    args = parser.parse_args()
    summary = run(args.output)
    print(f"BOEM: {summary['counts']['selected_wells']} API; k={summary['selected_k']}; silhouette={summary['silhouette']:.4f}")


if __name__ == "__main__":
    main()
