"""Кластеризация скважин (одна скважина = один объект) методом K-means."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "Depth_m", "Reservoir_Pressure_MPa", "Porosity_pct", "Permeability_mD",
    "Net_Pay_m", "Oil_Rate_bpd", "Gas_Rate_Mscf_d", "Water_Cut_pct",
]
TITLES = {
    "Depth_m": "Глубина, м",
    "Reservoir_Pressure_MPa": "Пластовое давление, МПа",
    "Porosity_pct": "Пористость, %",
    "Permeability_mD": "Проницаемость, мД",
    "Net_Pay_m": "Эффективная нефтенасыщенная толщина, м",
    "Oil_Rate_bpd": "Дебит нефти, барр./сут",
    "Gas_Rate_Mscf_d": "Дебит газа, тыс. станд. куб. футов/сут",
    "GOR_scf_bbl": "Газовый фактор, станд. куб. футов/барр.",
    "Water_Cut_pct": "Обводнённость, %",
}


@dataclass
class Result:
    wells: pd.DataFrame
    imputed: pd.DataFrame
    scaled: np.ndarray
    labels: np.ndarray
    scores: pd.DataFrame
    profiles: pd.DataFrame
    z_profiles: pd.DataFrame
    stability: pd.DataFrame
    quality: dict
    features: list[str]
    seed: int


def prepare_wells(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Проверка данных и медианная агрегация повторных измерений по Well_ID."""
    if len(features) < 2 or len(set(features)) != len(features):
        raise ValueError("Укажите минимум два разных числовых признака.")
    forbidden = [f for f in features if f.startswith("Future_") or f in
                 {"Well_ID", "Field_ID", "Timestamp", "cluster", "n_observations"}]
    if forbidden:
        raise ValueError(f"Идентификаторы и будущие значения не являются признаками: {forbidden}")
    missing = sorted(set(["Well_ID", *features]) - set(df.columns))
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {', '.join(missing)}")
    frame = df.copy()
    if frame.empty or frame["Well_ID"].isna().any():
        raise ValueError("Данные пусты или имеются пропуски Well_ID.")
    frame["Well_ID"] = frame["Well_ID"].astype(str).str.strip()
    if frame["Well_ID"].eq("").any():
        raise ValueError("Well_ID не может быть пустым.")
    for feature in features:
        try:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Нечисловые значения в {feature}.") from exc
        if np.isinf(frame[feature]).any():
            raise ValueError(f"Бесконечные значения в {feature}.")
        if feature in {*FEATURES, "GOR_scf_bbl"} and frame[feature].lt(0).any():
            raise ValueError(f"Отрицательные значения в {feature}.")
        if feature.endswith("_pct") and not frame[feature].dropna().between(0, 100).all():
            raise ValueError(f"Процент {feature} вне диапазона 0–100.")
    quality = {
        "n_rows": len(frame),
        "missing_raw": frame[features].isna().sum().astype(int).to_dict(),
        "excluded_future_columns": [c for c in frame if c.startswith("Future_")],
    }
    if "Timestamp" in frame:
        dates = pd.to_datetime(frame["Timestamp"], format="ISO8601", errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError("Timestamp содержит пропущенные или некорректные даты.")
        if pd.DataFrame({"id": frame["Well_ID"], "date": dates}).duplicated().any():
            raise ValueError("Повторяется пара Well_ID / Timestamp; проверьте дубли.")
        frame["Timestamp"] = dates
        quality["period_start"] = str(dates.min())
        quality["period_end"] = str(dates.max())
    if "Field_ID" in frame:
        frame["Field_ID"] = frame["Field_ID"].astype("string").str.strip().replace("", pd.NA)
        field_counts = frame.groupby("Well_ID")["Field_ID"].nunique()
        conflicts = field_counts[field_counts > 1].index.tolist()
        if conflicts:
            raise ValueError("Один Well_ID связан с несколькими Field_ID: "
                             + ", ".join(conflicts[:5])
                             + ". Используйте глобально уникальные идентификаторы скважин.")
    grouped = frame.groupby("Well_ID", sort=True)
    wells = grouped[features].median()
    if len(wells) < 3:
        raise ValueError("Для выбора k по силуэту нужны минимум три скважины.")
    if wells[features].isna().all(axis=1).any():
        raise ValueError("Есть скважины без единого наблюдаемого выбранного признака.")
    if wells[features].isna().all(axis=0).any():
        raise ValueError("Есть полностью пустые признаки; удалите их из --features.")
    wells["n_observations"] = grouped.size()
    if "Field_ID" in frame:
        quality["wells_with_multiple_fields"] = int((grouped["Field_ID"].nunique() > 1).sum())
        wells["Field_ID"] = grouped["Field_ID"].agg(
            lambda values: ", ".join(sorted(set(values.dropna().astype(str))))
        )
    if "Timestamp" in frame:
        wells["first_observation"] = grouped["Timestamp"].min()
        wells["last_observation"] = grouped["Timestamp"].max()
    quality["n_wells"] = len(wells)
    quality["missing_well_medians"] = wells[features].isna().sum().astype(int).to_dict()
    # Сравнение изменчивости внутри скважин с различиями их медиан.
    quality["within_well_std_median"] = grouped[features].std().median().to_dict()
    quality["between_well_median_std"] = wells[features].std().to_dict()
    return wells, quality


def analyze(df: pd.DataFrame, features: list[str] | None = None,
            max_k: int = 8, seed: int = 42, *,
            log_features: list[str] | None = None) -> Result:
    features = list(FEATURES if features is None else features)
    if max_k < 2:
        raise ValueError("--max-k должен быть не меньше 2.")
    if not 0 <= seed < 2**32 - 3:
        raise ValueError("--seed должен быть от 0 до 4294967292.")
    if log_features is None:
        log_features = [f for f in features if f == "Permeability_mD"]
    elif (not isinstance(log_features, list)
          or any(not isinstance(f, str) for f in log_features)
          or len(set(log_features)) != len(log_features)):
        raise ValueError("--log-features должен содержать список разных имен признаков.")
    else:
        log_features = list(log_features)
    unknown_log_features = sorted(set(log_features) - set(features))
    if unknown_log_features:
        raise ValueError("Признаки --log-features не входят в --features: "
                         + ", ".join(unknown_log_features))
    wells, quality = prepare_wells(df, features)
    for feature in log_features:
        if pd.to_numeric(df[feature], errors="raise").lt(0).any():
            raise ValueError(f"Для log1p нужны неотрицательные значения в {feature}.")
    imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(wells[features]),
                           index=wells.index, columns=features)
    transformed = imputed.copy()
    # Признаки с длинным правым хвостом задаются явно; профили сохраняют исходные единицы.
    for feature in log_features:
        transformed[feature] = np.log1p(transformed[feature])
    quality["log_features"] = log_features
    active = transformed.columns[transformed.nunique() > 1].tolist()
    quality["constant_features"] = [f for f in features if f not in active]
    if len(active) < 2:
        raise ValueError("После агрегации нужны минимум два непостоянных признака.")
    scaled = StandardScaler().fit_transform(transformed[active])
    max_valid_k = min(max_k, len(wells) - 1, len(np.unique(scaled, axis=0)))
    if max_valid_k < 2:
        raise ValueError("Недостаточно различных профилей скважин.")
    models, scores = {}, []
    for k in range(2, max_valid_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(scaled)
        if len(np.unique(model.labels_)) != k:
            continue
        models[k] = model
        scores.append({"k": k, "silhouette": silhouette_score(scaled, model.labels_),
                       "inertia": model.inertia_})
    if not scores:
        raise ValueError("Не удалось получить допустимое разбиение.")
    score_table = pd.DataFrame(scores)
    best_k = int(score_table.loc[score_table["silhouette"].idxmax(), "k"])
    model = models[best_k]
    # Номера удобны для чтения: по возрастанию среднего дебита (или первого признака).
    order_feature = "Oil_Rate_bpd" if "Oil_Rate_bpd" in features else features[0]
    order = imputed.assign(cluster=model.labels_).groupby("cluster")[order_feature].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order, start=1)}
    labels = np.array([mapping[label] for label in model.labels_])
    profiles = imputed.assign(cluster=labels).groupby("cluster").mean()
    profiles.insert(0, "n_wells", pd.Series(labels).value_counts().sort_index())
    z_profiles = pd.DataFrame(scaled, columns=active).assign(cluster=labels).groupby("cluster").mean()
    stability = []
    for other_seed in (seed + 1, seed + 2, seed + 3):
        other = KMeans(n_clusters=best_k, n_init=20, random_state=other_seed).fit_predict(scaled)
        stability.append({"seed": other_seed, "adjusted_rand_index": adjusted_rand_score(labels, other)})
    quality["active_features"] = active
    return Result(wells, imputed, scaled, labels, score_table, profiles, z_profiles,
                  pd.DataFrame(stability), quality, features, seed)


def markdown_table(frame: pd.DataFrame) -> str:
    """Markdown без дополнительной зависимости tabulate."""
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(map(fmt, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(map(fmt, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def report_text(result: Result, source: str) -> str:
    q, profiles = result.quality, result.profiles
    k = len(profiles)
    score = float(result.scores.loc[result.scores.k.eq(k), "silhouette"].iloc[0])
    lines = ["# Кластеризация скважин методом K-means", "", f"Источник: **{source}**.", "",
             f"Обработано {q['n_rows']} наблюдений по {q['n_wells']} скважинам. "
             f"Выбрано **k = {k}**, коэффициент силуэта **{score:.3f}**.", "",
             "## Подготовка и метод", "",
             "Одна строка модели — одна скважина. Повторные измерения агрегированы медианой "
             "за весь предоставленный период. Это характеристика типичного состояния, а не текущего режима. "
             "Пропуски игнорируются при агрегации; оставшиеся пустые медианы заменяются медианой по скважинам. "
             "Скважины без наблюдаемых выбранных признаков и полностью пустые признаки отклоняются.", "",
             "Признаки: " + ", ".join(f"`{f}`" for f in result.features) + ".", "",
             ("Преобразование log(1+x) применено к признакам: "
              + ", ".join(f"`{f}`" for f in q["log_features"]) + ". "
              if q["log_features"] else "Логарифмические преобразования не применялись. ")
             + "Затем StandardScaler приводит "
             "непостоянные признаки к нулевому среднему и единичной дисперсии. "
             "K-means работает в этом пространстве, PCA используется только для иллюстрации. "
             f"Для каждого k выполняется 20 инициализаций, random_state={result.seed}; "
             "выбирается максимальный средний силуэт среди проверенных k. Проверки наличия кластеров "
             "против модели с одной группой этот критерий не даёт.", "",
             "Идентификаторы, время и Future_* не участвуют в расстояниях. "
             "По умолчанию не включены параметры бурения и многочисленные взаимосвязанные показатели насыщенности, "
             "чтобы не смешивать слишком много разных процессов и не усиливать дублирующие признаки.", "",
             "## Качество данных", ""]
    if "period_start" in q:
        lines += [f"Период: {q['period_start']} — {q['period_end']}. "
                  "Проверьте происхождение данных и корректность дат перед содержательными выводами.", ""]
    counts = result.wells.n_observations
    lines += [f"Наблюдений на скважину: от {int(counts.min())} до {int(counts.max())}. "
              "Скважины имеют одинаковый вес независимо от числа наблюдений.", ""]
    missing = pd.DataFrame({"Признак": result.features,
                            "Пропусков в исходных строках": [q['missing_raw'][f] for f in result.features],
                            "Заполнено медиан скважин": [q['missing_well_medians'][f] for f in result.features]})
    lines += [markdown_table(missing), "",
              "Исключены будущие показатели: " + (", ".join(q["excluded_future_columns"]) or "нет") + ".", "",
              "Исключены постоянные признаки: " + (", ".join(q["constant_features"]) or "нет") + ".", ""]
    if q.get("wells_with_multiple_fields", 0):
        lines += ["Есть Well_ID с несколькими Field_ID. Проверьте глобальную уникальность идентификаторов.", ""]
    lines += ["## Выбор числа кластеров", "", markdown_table(result.scores), "",
              "![Выбор k](cluster_selection.png)", "", "## Профили и интерпретация", "",
              "В таблице — средние **медианных характеристик скважин** в исходных единицах; "
              "это не суммарная добыча и не средние по всем исходным строкам. "
              "Заполненные медианы включены в эти средние. Номера групп условны и не означают качество.", "",
              markdown_table(profiles.reset_index()), ""]
    for cluster, row in profiles.iterrows():
        z = result.z_profiles.loc[cluster]
        higher = [TITLES.get(f, f) for f in z.index if z[f] >= .5]
        lower = [TITLES.get(f, f) for f in z.index if z[f] <= -.5]
        lines += [f"### Кластер {cluster}: {int(row.n_wells)} скважин", "",
                  "Выше среднего по скважинам (не менее 0,5 стандартного отклонения): " +
                  ("; ".join(higher) or "нет выраженных признаков") + ".",
                  "Ниже среднего: " + ("; ".join(lower) or "нет выраженных признаков") + ".", ""]
    lines += ["Сравнения выше относятся к пространству после преобразований; это описательные "
              "пороги, а не статистические тесты значимости.", "", "![Профили](cluster_profiles.png)", "",
              "![PCA](clusters_pca.png)", "", "## Надёжность выводов", ""]
    if score < .25:
        lines += ["**Разделение слабое:** силуэт ниже 0,25 (ориентировочный, не универсальный порог). "
                  "Группы существенно перекрываются. Такое разбиение удобно как описательная сегментация, "
                  "но не подтверждает существование естественных типов скважин.", ""]
    else:
        lines += ["Силуэт характеризует компактность и разделение в выбранных признаках. "
                  "Даже высокий показатель не доказывает геологическую природу групп.", ""]
    lines += ["Устойчивость к трём другим начальным состояниям (ARI = 1 означает одинаковое разбиение):", "",
              markdown_table(result.stability), "",
              "Эта проверка касается инициализации алгоритма, а не устойчивости к выборке, "
              "периоду или набору признаков.", ""]
    if "Depth_m" in result.features:
        within = q["within_well_std_median"]["Depth_m"]
        between = q["between_well_median_std"]["Depth_m"]
        if np.isfinite(within) and np.isfinite(between) and within > between:
            lines += [f"Для глубины медианное стандартное отклонение внутри скважины — {within:.1f} м, "
                      f"а отклонение медиан между скважинами — {between:.1f} м. "
                      "Внутренняя изменчивость больше межскважинной: нужно уточнить смысл Depth_m, "
                      "проверить идентификаторы и обоснованность объединения всего периода.", ""]
    lines += ["K-means чувствителен к выбросам, выбору признаков и их корреляциям. "
              "Медианы скрывают временную динамику; перед эксплуатационными решениями повторите анализ "
              "на сопоставимом временном окне и проверьте результаты с геологом/технологом. "
              "Причины различий и эффективность мероприятий из кластеризации не следуют.", ""]
    return "\n".join(lines)


def save_outputs(result: Result, output: Path, source: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result.wells.assign(cluster=result.labels).to_csv(output / "well_clusters.csv", encoding="utf-8-sig")
    result.imputed.to_csv(output / "imputed_well_features.csv", encoding="utf-8-sig")
    result.profiles.to_csv(output / "cluster_profiles.csv", encoding="utf-8-sig")
    result.z_profiles.to_csv(output / "cluster_profiles_scaled.csv", encoding="utf-8-sig")
    result.scores.to_csv(output / "cluster_scores.csv", index=False)
    result.stability.to_csv(output / "stability.csv", index=False)
    # JSON null вместо нестандартного NaN для однократных измерений.
    quality = json.loads(pd.Series(result.quality).to_json(force_ascii=False))
    (output / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ["silhouette", "inertia"], ["Коэффициент силуэта", "Инерция (метод локтя)"]):
        ax.plot(result.scores.k, result.scores[metric], "o-")
        ax.axvline(len(result.profiles), color="tab:red", linestyle="--", alpha=.7)
        ax.set(xlabel="Число кластеров k", title=title, xticks=result.scores.k)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "cluster_selection.png", dpi=150)
    plt.close(fig)
    z = result.z_profiles
    fig, ax = plt.subplots(figsize=(max(9, len(z.columns) * 1.2), 4))
    bound = max(float(np.abs(z.to_numpy()).max()), .5)
    plot = ax.imshow(z, cmap="RdBu_r", aspect="auto", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(z.columns)), z.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(z)), [f"Кластер {i}" for i in z.index])
    ax.set_title("Профили кластеров: отклонение от среднего в единицах σ")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "cluster_profiles.png", dpi=150)
    plt.close(fig)
    pca = PCA(n_components=2).fit(result.scaled)
    xy = pca.transform(result.scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in result.profiles.index:
        mask = result.labels == cluster
        ax.scatter(xy[mask, 0], xy[mask, 1], label=f"Кластер {cluster} (n={mask.sum()})", alpha=.8)
    variance = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})",
           title=f"Скважины в проекции PCA: {variance.sum():.1%} дисперсии")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "clusters_pca.png", dpi=150)
    plt.close(fig)
    (output / "report.md").write_text(report_text(result, source), encoding="utf-8")


def make_demo(seed: int = 42) -> pd.DataFrame:
    """Искусственно разделённые группы; не модель конкретного месторождения."""
    rng = np.random.default_rng(seed)
    centers = np.array([[2200, 24, 14, 35, 12, 180, 600, 62],
                        [2800, 32, 21, 140, 25, 650, 1100, 28],
                        [3500, 42, 26, 320, 40, 1100, 1700, 10]])
    spread = np.array([130, 2, 1.2, 12, 2, 45, 65, 3])
    rows = []
    for i in range(90):
        base = centers[i // 30] + rng.normal(size=8) * spread
        for day in range(4):
            values = np.maximum(base + rng.normal(size=8) * spread * .2, .01)
            row = dict(zip(FEATURES, values.round(3)))
            row.update(Well_ID=f"DEMO_{i + 1:03d}", Timestamp=f"2025-01-{day + 1:02d}", Field_ID="DEMO")
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Путь к CSV со столбцом Well_ID")
    source.add_argument("--demo", action="store_true", help="Синтетический учебный пример")
    parser.add_argument("--output", type=Path, default=Path("results/custom"))
    parser.add_argument("--features", nargs="+", default=FEATURES)
    parser.add_argument("--log-features", nargs="*", default=None,
                        help="Признаки для log1p; по умолчанию Permeability_mD, если выбран. "
                             "Пустой список отключает log1p.")
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sep", default=",", help="Разделитель CSV")
    parser.add_argument("--decimal", default=".", help="Десятичный разделитель")
    args = parser.parse_args()
    try:
        df = make_demo() if args.demo else pd.read_csv(args.input, sep=args.sep, decimal=args.decimal,
                                                       dtype={"Well_ID": "string", "Field_ID": "string"})
        result = analyze(df, args.features, args.max_k, args.seed, log_features=args.log_features)
        save_outputs(result, args.output, "СИНТЕТИЧЕСКИЕ УЧЕБНЫЕ ДАННЫЕ" if args.demo else args.input.name)
        if args.demo:
            df.to_csv(args.output / "synthetic_wells.csv", index=False)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    chosen = result.scores.loc[result.scores.k.eq(len(result.profiles))].iloc[0]
    print(f"Скважин: {len(result.wells)}; k={int(chosen.k)}; силуэт={chosen.silhouette:.3f}")
    print(f"Отчёт: {(args.output / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
