"""Build the DOCX and Markdown report from the analysis summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]


def number(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def report_blocks(summary):
    counts = summary["counts"]
    clusters = summary["clusters"]
    stability = summary["stability"]
    source = summary["source"]
    k_values = [row["k"] for row in summary["k_scores"]]
    n = number(counts["selected_wells"])
    k = summary["selected_k"]
    blocks = []

    def add(kind, text=None, **kwargs):
        blocks.append({"kind": kind, "text": text, **kwargs})

    add("title", "Кластеризация нефтяных скважин по открытым данным BOEM")
    add("subtitle", "Реферат по теме 9 «Кластеризация данных»\nПредмет «Анализ данных в нефтегазе 26/27»")
    add("h1", "1 Задача и основные методы")
    add("p", f"В работе скважины группируются по показателям добычи за {source['period']}. "
        f"После отбора осталось {n} скважин. Для K-means выбрано число групп k={k}. "
        f"Средний коэффициент силуэта равен {number(summary['silhouette'], 3)}. "
        "Практическая задача состоит в том, чтобы найти скважины с похожими эксплуатационными "
        "характеристиками и определить, какие дополнительные данные нужны для проверки различий между группами.")
    add("p", "Кластеризация относится к обучению без учителя: готовых меток типов скважин "
        "в исходных данных нет. Близость определяется выбранными признаками и расстоянием между "
        "их значениями. Поэтому один фонд можно разделить по-разному, если изучать геологию, "
        "добычу или техническое состояние оборудования. В этом примере рассматривается эксплуатационный профиль [2].")
    add("p", "K-means относит каждый объект к ближайшему центру и пересчитывает центры как средние "
        "значения признаков. Число групп k задаётся до обучения. Метод удобен для компактных групп "
        "скважин по дебиту, обводнённости и газовому фактору. Он чувствителен к масштабу признаков "
        "и выбросам, а группы сложной формы может описывать плохо [2].")
    add("formula", "J = Σⱼ₌₁ᵏ Σᵢ∈Cⱼ ‖zᵢ − μⱼ‖² → min")
    add("p", "Здесь zᵢ обозначает преобразованный вектор признаков скважины i, Cⱼ соответствует "
        "кластеру j, а μⱼ является его центром. Минимизируется сумма квадратов евклидовых "
        "расстояний до центров. Алгоритм повторяет назначение объектов и пересчёт центров до сходимости. "
        "Он может прийти к локальному минимуму, поэтому используются разные начальные центры.")
    add("p", "Иерархическая кластеризация Ward последовательно объединяет группы с минимальным "
        "приростом внутригрупповой суммы квадратов. Дендрограмма позволяет рассматривать фонд "
        "с разной детализацией, например разделить общую группу обводнённых скважин на подгруппы. "
        "Для крупных выборок такой расчёт требует больше памяти. Ward также зависит от масштаба признаков [3].")
    add("p", "DBSCAN объединяет соседние объекты в плотных областях, а изолированные точки "
        "помечает как шум. Это полезно для поиска необычных режимов скважин. Число групп заранее "
        "не задаётся, но необходимо выбрать радиус соседства eps и минимальную численность "
        "окрестности min_samples, включая сам объект. Разная плотность групп осложняет подбор параметров [4].")
    add("p", "Смесь гауссовых распределений GMM описывает данные несколькими вероятностными "
        "компонентами и оценивает вероятность принадлежности скважины каждой из них в рамках модели. "
        "Так можно показать неопределённость для переходных режимов. Модель допускает "
        "эллипсоидальные группы, но зависит от предположений о распределениях и числе компонент [5].")

    add("pagebreak")
    add("h1", "2 Открытые данные и подготовка признаков")
    add("p", f"Источник: {source['name']} [1]. Период анализа: {source['period']}. "
        f"Исходный набор содержит {number(counts['raw_rows'])} строк и "
        f"{number(counts['raw_wells'])} уникальных идентификаторов скважин. "
        "Опубликованные сведения о добыче позволяют проверить расчёты и повторить отбор. "
        "Используется сохранённая копия выгрузки, поскольку исходная публикация может обновляться. "
        "BOEM разрешает копирование публичной информации с указанием источника, "
        "но не гарантирует полноту и безошибочность данных [11].")
    add("p", "Единица анализа соответствует стволу скважины с 12-значным номером API, "
        "который включает код бокового ствола. Исходная запись описывает месяц работы "
        "объекта эксплуатации (completion). Записи объединяются в годовой профиль API. "
        "Сам идентификатор не участвует в вычислении расстояний [9].")
    add("table", headers=["Условие отбора", "API до", "API после"], rows=[
        [item["label"], number(item["before"]), number(item["after"])]
        for item in summary["filters"]], widths=[11.0, 2.5, 2.5])
    add("p", f"После отбора в анализ включены {n} скважин "
        f"и {number(counts['selected_rows'])} исходных строк. "
        "Единственный активный объект эксплуатации исключает двойной счёт дней работы "
        "при параллельной добыче. Фильтры ограничивают выводы регулярно работающим нефтяным фондом. "
        "Каждая скважина имеет одинаковый вес при обучении независимо от объёма добычи.")
    add("small", "Стволы с ненулевой зарегистрированной закачкой исключены: Days Produced "
        "может учитывать дни закачки. Для газлифтных скважин инструкция требует указывать "
        "пластовый газ за вычетом газа, поданного на газлифт, поэтому высокий GOR нельзя "
        "объяснять прямым включением газлифтного газа в добычу [10].")
    add("table", headers=["Признак", "Расчёт и единицы"], rows=[
        [item["label"], f"{item['formula']}. {item['unit']}"]
        for item in summary["features"]], widths=[5.0, 11.0])
    add("p", "Нефтяной дебит и газовый фактор преобразуются функцией log(1+x), которая "
        "сжимает большой диапазон значений. Затем все признаки стандартизируются: из каждого "
        "значения вычитается среднее по скважинам, результат делится на стандартное отклонение. "
        "Так признак с численно большим масштабом не получает преимущество только из-за единиц "
        "измерения. При этом StandardScaler не устраняет влияние выбросов [6].")
    add("pagebreak")
    add("h1", "3 Выбор числа групп и результаты")
    add("p", f"Проверены значения k от {min(k_values)} до {max(k_values)}. "
        "Для каждого выполняются 20 инициализаций K-means при random_state=42. "
        "Из них берётся запуск с минимальной инерцией, то есть целевой суммой квадратов. "
        "Затем значения k сравниваются по среднему силуэту и выбирается его максимум. Для отдельной скважины "
        "s=(b−a)/max(a,b), где a является средним расстоянием до остальных объектов своей группы, "
        "а b является наименьшим средним расстоянием до объектов другого кластера. Значения около нуля указывают на "
        "перекрытие, отрицательные значения требуют проверки назначения [7].")

    add("p", f"По выбранному критерию получено k={k}, силуэт равен "
        f"{number(summary['silhouette'], 3)}. Профили групп описаны медианами исходных "
        "характеристик входящих в них скважин. Они выражены в физических единицах "
        "и отличаются от центров K-means в преобразованном пространстве.")
    add("table", headers=["Группа", "Скважин", "Нефть\nбарр./сут работы", "Вода\n%", "GOR\nscf/bbl"], rows=[
        [str(item["cluster"]), number(item["n_wells"]), number(item["oil_rate_bpd"], 1),
         number(item["water_cut_pct"], 1), number(item["gor_scf_bbl"], 0)]
        for item in clusters], widths=[2.1, 2.3, 3.7, 3.1, 4.8])
    for item in clusters:
        if item["label"] != f"Группа {item['cluster']}":
            add("p", f"Группа {item['cluster']}: {item['label']}.")
    if "negative_silhouette_fraction" in summary:
        add("p", f"У {number(100 * summary['negative_silhouette_fraction'], 1)}% скважин "
            "индивидуальный силуэт отрицателен: по выбранным расстояниям они в среднем "
            "ближе к соседней группе. При дальнейшем изучении фонда такие пограничные "
            "назначения стоит проверять отдельно.")
    add("p", "Для каждой группы выбран ближайший к её центру объект в пространстве модели. Значения "
        "относятся к этим скважинам, а не ко всей группе.")
    add("table", headers=["Идентификатор", "Группа", "Нефть\nбарр./сут работы", "Вода\n%", "GOR\nscf/bbl"], rows=[
        [str(item["well_id"]), str(item["cluster"]), number(item["oil_rate_bpd"], 1),
         number(item["water_cut_pct"], 1), number(item["gor_scf_bbl"], 0)]
        for item in summary["examples"]], widths=[4.2, 1.8, 3.5, 2.5, 4.0])
    wet = max(clusters, key=lambda item: item["water_cut_pct"])
    oil = max(clusters, key=lambda item: item["oil_rate_bpd"])
    no_gor = next(check for check in stability["checks"] if check["name"] == "Без газового фактора")
    add("p", "Группы прежде всего различаются дебитом нефти и обводнённостью. "
        f"При исключении GOR состав меняется мало: ARI={number(no_gor['ari'], 3)}. "
        f"Для группы {wet['cluster']} с наибольшей медианной обводнённостью "
        "полезно проверить динамику поступления воды и ограничения оборудования по обработке "
        f"жидкости. Группа {oil['cluster']} с наибольшим медианным дебитом нефти "
        "подходит для подбора эксплуатационных аналогов при сопоставимых условиях.")
    add("p", "Из кластеризации нельзя установить причину обводнения или рассчитать эффект "
        "ремонта. Для этого потребуются история работы, конструкция скважины, свойства "
        "пласта и результаты исследований. Номера групп условны и не являются оценкой качества скважин.")
    variance = sum(summary["pca"]["explained_variance_ratio"][:2])
    add("p", f"Две компоненты PCA сохраняют {number(100 * variance, 1)}% дисперсии "
        "преобразованных признаков. PCA применяется только для изображения скважин "
        "на плоскости. Само разбиение рассчитано по всем выбранным признакам.")

    add("pagebreak")
    add("h1", "4 Устойчивость и ограничения выводов")
    seed_result = (f"При смене начальных состояний ARI равен {number(stability['seed_ari_min'], 3)} "
                   "во всех проверенных запусках. "
                   if stability["seed_ari_min"] == stability["seed_ari_max"] else
                   f"При смене начальных состояний ARI изменяется от {number(stability['seed_ari_min'], 3)} "
                   f"до {number(stability['seed_ari_max'], 3)}. ")
    add("p", seed_result + "ARI сравнивает назначения объектов с поправкой на случайное совпадение и не зависит "
        "от нумерации кластеров. Значение 1 соответствует одинаковому разбиению. Такая проверка "
        "показывает чувствительность к инициализации, но сама по себе не подтверждает "
        "геологическую природу групп [8].")
    add("p", f"При том же числе групп метод Ward даёт силуэт "
        f"{number(stability['ward_silhouette'], 3)} и ARI "
        f"{number(stability['ward_ari'], 3)} относительно K-means.")
    if "subsample_ari_median" in stability:
        add("small", f"При повторном анализе подвыборок медиана ARI составляет "
            f"{number(stability['subsample_ari_median'], 3)}, диапазон "
            f"{number(stability['subsample_ari_min'], 3)}…{number(stability['subsample_ari_max'], 3)}. "
            "Масштабирование и модель обучаются заново, сравнение выполняется на общих объектах.")
    for check in stability.get("checks", []):
        add("small", f"{check['name']}: ARI={number(check['ari'], 3)}. {check['note']}")
    for caveat in summary["caveats"]:
        add("small", caveat)
    robust = next((check for check in stability.get("checks", []) if check["name"] == "RobustScaler"), None)
    if robust is not None:
        add("p", "При фиксированной подготовке данных группы устойчивы к инициализации "
            "и проверенным подвыборкам. Замена StandardScaler на RobustScaler существенно "
            f"меняет назначения: ARI={number(robust['ari'], 3)}. Поэтому результат описывает "
            "выбранный фонд в рамках этой методики. Он помогает искать эксплуатационные аналоги "
            "и выбирать объекты для проверки, но не устанавливает универсальные типы скважин.")
    add("small", f"Повторить расчёт из корня репозитория: {summary['reproduce_command']}. "
        "Паспорт исходных данных и полные таблицы отбора находятся в data/boem, "
        "результаты и параметры запуска сохраняются в results/boem_2023.")
    add("h1", "Источники")
    sources = [
        (f"{source['name']}. Дата обращения {source['accessed_at']}.", source["url"]),
        ("Scikit-learn. KMeans.", "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html"),
        ("Scikit-learn. AgglomerativeClustering.", "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html"),
        ("Scikit-learn. DBSCAN.", "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"),
        ("Scikit-learn. Gaussian mixture models.", "https://scikit-learn.org/stable/modules/mixture.html"),
        ("Scikit-learn. StandardScaler.", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"),
        ("Scikit-learn. Silhouette coefficient.", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html"),
        ("Scikit-learn. Adjusted Rand index.", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html"),
        ("BOEM. Field Definitions for OGORA.", "https://www.data.boem.gov/Main/HtmlPage.aspx?page=ogorA"),
        ("ONRR. Minerals Production Reporter Handbook. Appendix H, Well Codes.", "https://onrr.gov/document/MPRH-Appendix-H.pdf"),
        ("BOEM. Disclaimer of Liability and Endorsement.", source["license_or_terms"]),
    ]
    for i, (label, url) in enumerate(sources, 1):
        add("source", f"{i}. {label}", url=url)
    return blocks


def set_font(style, name, size, bold=False):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in list(fonts.attrib):
        if "theme" in attr.lower():
            del fonts.attrib[attr]
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{attr}"), name)
    ppr = style.element.get_or_add_pPr()
    for child in list(ppr):
        if child.tag in {qn("w:pBdr"), qn("w:numPr")}:
            ppr.remove(child)
    for child in list(rpr):
        if child.tag in {qn("w:spacing"), qn("w:kern"), qn("w:szCs")}:
            rpr.remove(child)


def hyperlink(paragraph, label, url):
    rel = paragraph.part.relate_to(url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1D465F")
    rpr.append(color)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "19")
    rpr.append(size)
    run.append(rpr)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def add_table(document, block):
    table = document.add_table(rows=1, cols=len(block["headers"]))
    table.autofit = False
    for i, width in enumerate(block["widths"]):
        table.columns[i].width = Cm(width)
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        for attr, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            el.set(qn(f"w:{attr}"), value)
        borders.append(el)
    tbl_pr.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), "85")
        el.set(qn("w:type"), "dxa")
        margins.append(el)
    tbl_pr.append(margins)
    for row_idx, values in enumerate([block["headers"], *block["rows"]]):
        row = table.rows[0] if row_idx == 0 else table.add_row()
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        if row_idx == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
        for col_idx, value in enumerate(values):
            cell = row.cells[col_idx]
            cell.width = Cm(block["widths"][col_idx])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = str(value)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            for run in p.runs:
                run.font.size = Pt(10.5)
                run.bold = row_idx == 0
            if row_idx == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "E8EEF2")
                cell._tc.get_or_add_tcPr().append(shade)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def add_objective_equation(paragraph):
    """Editable Office Math with real summation limits and subscripts."""
    from lxml import etree

    equation = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
      <m:oMath>
        <m:r><m:t>J = </m:t></m:r>
        <m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
          <m:sub><m:r><m:t>j=1</m:t></m:r></m:sub>
          <m:sup><m:r><m:t>k</m:t></m:r></m:sup>
          <m:e><m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/><m:supHide m:val="1"/></m:naryPr>
            <m:sub><m:r><m:t>i∈</m:t></m:r><m:sSub><m:e><m:r><m:t>C</m:t></m:r></m:e><m:sub><m:r><m:t>j</m:t></m:r></m:sub></m:sSub></m:sub>
            <m:sup/>
            <m:e><m:sSup><m:e><m:d><m:dPr><m:begChr m:val="‖"/><m:endChr m:val="‖"/></m:dPr>
              <m:e><m:sSub><m:e><m:r><m:t>z</m:t></m:r></m:e><m:sub><m:r><m:t>i</m:t></m:r></m:sub></m:sSub>
                <m:r><m:t>−</m:t></m:r>
                <m:sSub><m:e><m:r><m:t>μ</m:t></m:r></m:e><m:sub><m:r><m:t>j</m:t></m:r></m:sub></m:sSub>
              </m:e></m:d></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup></m:e>
          </m:nary></m:e>
        </m:nary>
        <m:r><m:t> → min</m:t></m:r>
      </m:oMath>
    </m:oMathPara>"""
    paragraph._p.append(etree.fromstring(equation.encode("utf-8")))


def build_docx(blocks, output):
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.8)
    section.left_margin = section.right_margin = Cm(2.5)
    styles = document.styles
    for style_name, size, bold in [("Normal", 11.5, False), ("Title", 18, True),
                                   ("Subtitle", 11, False), ("Heading 1", 13, True)]:
        set_font(styles[style_name], "Times New Roman", size, bold)
    normal = styles["Normal"].paragraph_format
    normal.space_after = Pt(6)
    normal.line_spacing = 1.08
    normal.widow_control = True
    styles["Heading 1"].paragraph_format.space_before = Pt(9)
    styles["Heading 1"].paragraph_format.space_after = Pt(7)
    styles["Title"].paragraph_format.space_after = Pt(8)
    styles["Subtitle"].paragraph_format.space_after = Pt(9)
    for block in blocks:
        kind = block["kind"]
        if kind == "pagebreak":
            document.add_page_break()
        elif kind == "table":
            add_table(document, block)
        elif kind == "source":
            p = document.add_paragraph()
            p.paragraph_format.space_after = Pt(3)
            hyperlink(p, block["text"], block["url"])
            for run in p.runs:
                run.font.size = Pt(9.5)
        else:
            style = {"title": "Title", "subtitle": "Subtitle", "h1": "Heading 1"}.get(kind, "Normal")
            p = document.add_paragraph(block["text"], style=style)
            if kind == "formula":
                p.clear()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next = True
                add_objective_equation(p)
            if kind == "small":
                p.paragraph_format.space_after = Pt(4)
                for run in p.runs:
                    run.font.size = Pt(10.5)
    document.core_properties.title = "Кластеризация нефтяных скважин по открытым данным BOEM"
    document.core_properties.subject = "Анализ данных в нефтегазе 26/27"
    document.core_properties.author = ""
    document.core_properties.last_modified_by = ""
    document.core_properties.comments = ""
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def build_markdown(blocks, output):
    lines = []
    for block in blocks:
        kind = block["kind"]
        if kind == "pagebreak":
            continue
        if kind == "table":
            escape = lambda value: str(value).replace("\n", " ").replace("|", "\\|")
            lines.append("| " + " | ".join(map(escape, block["headers"])) + " |")
            lines.append("| " + " | ".join("---" for _ in block["headers"]) + " |")
            lines.extend("| " + " | ".join(map(escape, row)) + " |" for row in block["rows"])
        elif kind == "source":
            lines.append(f"[{block['text']}]({block['url']})")
        else:
            prefix = {"title": "# ", "h1": "## "}.get(kind, "")
            lines.append(prefix + block["text"])
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results/boem_2023/summary.json")
    parser.add_argument("--docx", type=Path, default=ROOT / "report.docx")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/report.md")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    blocks = report_blocks(summary)
    build_docx(blocks, args.docx)
    build_markdown(blocks, args.markdown)
    print(f"Saved {args.docx}\nSaved {args.markdown}")


if __name__ == "__main__":
    main()
