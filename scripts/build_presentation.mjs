#!/usr/bin/env node
/**
 * Editable presentation from the analysis outputs.
 * Requires the Codex bundled @oai/artifact-tool runtime.
 * RUNTIME_NODE_MODULES points to its node_modules directory.
 * --finalize also requires PRESENTATION_SKILL_DIR and RUNTIME_PYTHON.
 * Use a new --output path with --finalize; validation never overwrites a file.
 * All research values are read from summary.json; none are copied by hand.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { values: args } = parseArgs({ options: {
  input: { type: 'string', default: path.join(root, 'results/boem_2023/summary.json') },
  output: { type: 'string', default: path.join(root, 'presentation.pptx') },
  'render-dir': { type: 'string' },
  'notes-output': { type: 'string', default: path.join(root, 'docs/speaker_notes.md') },
  finalize: { type: 'boolean', default: false },
} });
const runtimeModules = process.env.RUNTIME_NODE_MODULES || process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
const runtimeModule = runtimeModules
  ? pathToFileURL(path.join(runtimeModules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href
  : '@oai/artifact-tool';
const { Presentation, PresentationFile, FileBlob } = await import(runtimeModule);
const summary = JSON.parse(await fs.readFile(path.resolve(args.input), 'utf8'));

const C = { paper: '#F7F6F0', ink: '#183A3D', muted: '#536769', teal: '#087F8C',
  ochre: '#C77C24', blue: '#4966A2', grid: '#D9DFD9', pale: '#E9EDE7', white: '#FFFFFF' };
const palette = [C.teal, C.ochre, C.blue, '#8E567C', '#577047', '#C6533C', '#68748A', '#806A41'];
const font = 'Arial';
const W = 1280, H = 720;
const deck = Presentation.create({ slideSize: { width: W, height: H } });
const talk = [];
const nfmt = (value, digits = 0) => Number(value).toLocaleString('ru-RU', {
  minimumFractionDigits: digits, maximumFractionDigits: digits,
});
// Native chart workbooks use Excel numeric cells (15 significant digits).
// Ten significant digits preserve plot geometry without unrepresentable literals.
// The source summary.json remains at full numerical precision.
const chartNumber = value => Number(Number(value).toPrecision(10));
const wellWord = n => n % 100 >= 11 && n % 100 <= 14 ? 'стволов' : n % 10 === 1 ? 'ствол' : [2, 3, 4].includes(n % 10) ? 'ствола' : 'стволов';
const groupWord = n => n % 100 >= 11 && n % 100 <= 14 ? 'групп' : n % 10 === 1 ? 'группа' : [2, 3, 4].includes(n % 10) ? 'группы' : 'групп';

function text(slide, content, x, y, w, h, size = 28, options = {}) {
  const shape = slide.shapes.add({ geometry: 'textbox', name: options.name || content.slice(0, 45),
    position: { left: x, top: y, width: w, height: h }, fill: 'none',
    line: { fill: 'none', width: 0 } });
  shape.text = String(content);
  shape.text.insets = 0;
  shape.text.style = { fontFamily: font, fontSize: size, color: options.color || C.ink,
    bold: options.bold || false, verticalAlignment: 'top', ...options };
  return shape;
}

function slide(title, notes) {
  const s = deck.slides.add();
  s.background.fill = C.paper;
  if (title) text(s, title, 64, 43, 1152, 80, 46, { bold: true });
  const number = deck.slides.items.length;
  text(s, String(number).padStart(2, '0'), 1164, 672, 52, 24, 18, { color: C.muted, alignment: 'right' });
  s.speakerNotes.text = notes;
  talk.push({ title: title || 'Кластеризация добывающих скважин', notes });
  return s;
}

function footer(s, content) { text(s, content, 64, 647, 1060, 45, 18, { color: C.muted }); }

function table(s, values, { x = 64, y = 170, w = 1152, h = 360, widths, size = 24 } = {}) {
  const t = s.tables.add({ rows: values.length, columns: values[0].length,
    left: x, top: y, width: w, height: h, values,
    ...(widths ? { columnWidths: widths } : {}) });
  t.styleOptions = { headerRow: false, bandedRows: false };
  t.borders.assign({ fill: C.paper, width: 1.2, style: 'solid' });
  for (let row = 0; row < values.length; row++) {
    for (let col = 0; col < values[0].length; col++) {
      const cell = t.getCell(row, col);
      cell.fill = row === 0 ? C.ink : (row % 2 ? C.pale : C.paper);
      cell.text.style = { fontFamily: font, fontSize: size,
        color: row === 0 ? C.white : C.ink, bold: row === 0 };
    }
  }
  return t;
}

const chartText = { typeface: font, fontSize: 23, fill: C.muted };
function axis(title, more = {}) {
  return { visible: true, title: { text: title, textStyle: { ...chartText, fontSize: 23 } },
    textStyle: chartText, line: { fill: C.grid, width: 1 },
    majorGridlines: { fill: C.grid, width: 1 }, ...more };
}
function chart(s, type, options) {
  return s.charts.add(type, { chartFill: C.paper, plotAreaFill: C.paper,
    chartLine: { fill: 'none', width: 0 }, plotAreaLine: { fill: 'none', width: 0 },
    titleTextStyle: { ...chartText, fontSize: 25, bold: true },
    legend: { position: 'bottom', overlay: false, textStyle: { ...chartText, fontSize: 22 } },
    dataLabels: { textStyle: { ...chartText, fontSize: 22 } }, ...options });
}

// The following renderer is populated after the analysis has written its
// documented summary schema. An absent field is an error, never a fake value.
function requireField(object, key) {
  if (object[key] === undefined || object[key] === null) throw new Error(`Missing summary field: ${key}`);
  return object[key];
}

async function writeOutputs() {
  const outputPath = path.resolve(args.output);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  let renderDeck = deck;
  let receiptPath;
  if (args.finalize) {
    const skill = process.env.PRESENTATION_SKILL_DIR;
    const python = process.env.RUNTIME_PYTHON;
    if (!skill || !python) throw new Error('--finalize requires PRESENTATION_SKILL_DIR and RUNTIME_PYTHON.');
    const { finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
    const staging = path.join(root, '.build', 'presentation', path.basename(outputPath, '.pptx'));
    await fs.mkdir(staging, { recursive: true });
    const candidatePath = path.join(staging, 'candidate.pptx');
    receiptPath = path.join(staging, 'validation.json');
    await (await PresentationFile.exportPptx(deck)).save(candidatePath);
    const tables = [2, 3, 4, 7, 8, 9];
    await finalizePresentation({
      workspaceDir: root, candidatePath, finalPath: outputPath,
      pythonExecutable: python,
      integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
      layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
      layoutArgs: ['--expected-slide-size-emu', '12192000,6858000',
        '--validate-bullet-geometry', '--validate-heading-fit',
        ...tables.flatMap(n => ['--require-native-table-slide', String(n)])],
      explicitTotalSlideCount: 10,
      requiredNativeTableOwnerSlides: tables,
      requiredNativeChartOwnerSlides: [5, 6],
      materializeLiteralChartWorkbooks: true,
      fontPolicy: { basis: 'design', families: [font] },
      verifyArtifactToolImport: true,
      receiptPath,
    });
    renderDeck = await PresentationFile.importPptx(await FileBlob.load(outputPath));
  } else {
    await (await PresentationFile.exportPptx(deck)).save(outputPath);
  }
  await fs.mkdir(path.dirname(path.resolve(args['notes-output'])), { recursive: true });
  await fs.writeFile(path.resolve(args['notes-output']), '# Текст выступления\n\n' + talk.map((item, i) =>
    `## ${i + 1}. ${item.title}\n\n${item.notes}\n`).join('\n'), 'utf8');
  if (args['render-dir']) {
    const output = path.resolve(args['render-dir']);
    await fs.mkdir(output, { recursive: true });
    for (let i = 0; i < renderDeck.slides.items.length; i++) {
      const blob = await renderDeck.export({ slide: renderDeck.slides.items[i], format: 'png', scale: 1 });
      await fs.writeFile(path.join(output, `slide-${String(i + 1).padStart(2, '0')}.png`),
        new Uint8Array(await blob.arrayBuffer()));
    }
  }
  console.log(JSON.stringify({ output: outputPath, slides: deck.slides.items.length, receipt: receiptPath }));
}

function methodSlide() {
  const notes = 'Кластеризация объединяет похожие объекты без заранее известных меток. ' +
    'В нашем случае объектом является ствол скважины, а признаки описывают его добычу. ' +
    'K-means минимизирует сумму квадратов расстояний до центров. Он удобен как простой ' +
    'и объяснимый способ сегментации фонда, но предпочитает компактные группы и зависит ' +
    'от масштаба признаков. Иерархический метод Ward последовательно объединяет группы ' +
    'с наименьшим приростом внутригрупповой суммы квадратов. DBSCAN выделяет плотные ' +
    'области и шум, однако его результат зависит от радиуса окрестности и плотности. ' +
    'Гауссова смесь описывает перекрывающиеся компоненты и даёт мягкие вероятностные ' +
    'принадлежности. Эти вероятности относятся к модели, а не к геологической истинности группы.\n\n' +
    'Источники: https://scikit-learn.org/stable/modules/clustering.html ; ' +
    'https://scikit-learn.org/stable/modules/mixture.html';
  const s = slide('Основные методы кластеризации', notes);
  text(s, 'Выбор метода зависит от формы групп и задачи анализа', 64, 127, 1152, 42, 28);
  table(s, [
    ['Метод', 'Как строится группа', 'Применение и ограничение'],
    ['K-means', 'Объекты ближе к одному центру', 'Обзор фонда по дебитам. Важны масштаб и выбросы.'],
    ['Иерархический\nметод Ward', 'Близкие группы последовательно объединяются', 'Поиск аналогов на разных уровнях. Дерево зависит от признаков.'],
    ['DBSCAN', 'Связанные плотные области; редкие точки считаются шумом', 'Поиск необычных режимов. Нужно выбрать радиус и плотность.'],
    ['Гауссова смесь', 'Вероятность принадлежности каждой компоненте', 'Перекрывающиеся режимы. Нужно проверить форму распределений.'],
  ], { y: 192, h: 420, widths: [225, 435, 492], size: 24 });
  footer(s, 'В практической части: K-means и сравнение с Ward');
}

function featuresSlide(source) {
  const notes = 'Для каждого ствола рассчитываем три годовые характеристики. ' +
    'Дебит нефти равен сумме добытой нефти, делённой на суммарное число активных дней. ' +
    'Обводнённость рассчитываем как долю воды в сумме нефти и воды за тот же период. ' +
    'Газовый фактор равен суммарному газу, делённому на суммарную нефть. Газ в OGOR-A ' +
    'задан в тысячах кубических футов, поэтому в формуле присутствует множитель 1000. ' +
    'Мы сначала суммируем объёмы, а затем делим. Простое среднее месячных отношений ' +
    'дало бы одинаковый вес месяцам с очень разными объёмами. ' +
    'Для дебита и газового фактора применяем log1p, чтобы уменьшить влияние длинных ' +
    'правых хвостов. После этого стандартизируем каждый признак. K-means работает ' +
    'с евклидовыми расстояниями в полученном трёхмерном пространстве. При фиксированном ' +
    'k алгоритм чередует отнесение к ближайшему центру и пересчёт средних центров. ' +
    'Из 20 запусков выбирается решение с минимальной инерцией.\n\n' +
    `Данные: ${source}\n` +
    'Метод: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html ; ' +
    'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html';
  const s = slide('Три характеристики одного ствола', notes);
  table(s, [
    ['Признак', 'Расчёт за 2023 год', 'Единица'],
    ['Дебит нефти qₒ', 'Σ нефть / Σ активные дни', 'барр./сут работы'],
    ['Обводнённость WC', '100 × Σ вода / Σ (нефть + вода)', '%'],
    ['Газовый фактор GOR', '1000 × Σ газ (Mcf) / Σ нефть', 'ст. фут³/барр.'],
  ], { y: 162, h: 284, widths: [340, 500, 312], size: 26 });
  text(s, 'log1p для qₒ и GOR, затем стандартизация', 64, 475, 1152, 42, 30, { bold: true, color: C.teal });
  text(s, 'Из значения вычитаем среднее и делим на стандартное отклонение.', 64, 524, 1145, 40, 26);
  text(s, 'K-means: ближайший центр и пересчёт центров до сходимости.\n20 инициализаций для каждого k, random_state = 42.', 64, 574, 1145, 68, 25);
  footer(s, 'Идентификатор ствола и дата не участвуют в расчёте расстояний');
}

function cover(d) {
  const s = slide('', `Мы рассматриваем кластеризацию добывающих скважин на реальных открытых данных ` +
    `${d.source.name}. Единицей анализа является 12-значный API, то есть отдельный ствол с учётом ` +
    `бокового ответвления. После отбора осталось ${d.counts.selected_wells} стволов. ` +
    `Мы хотим найти группы с похожими годовым дебитом нефти, обводнённостью и газовым фактором. ` +
    `Практический результат помогает выбрать объекты для сравнения и дальнейшего изучения. ` +
    `Он не определяет причину отличий и не назначает ремонт.\n\nДанные: ${d.source.url}`);
  text(s, 'Анализ данных в нефтегазе 26/27', 64, 63, 1100, 40, 26, { color: C.muted });
  text(s, 'Кластеризация\nдобывающих скважин', 64, 164, 810, 190, 62, { bold: true });
  text(s, nfmt(d.counts.selected_wells), 910, 188, 305, 110, 88, { bold: true, color: C.teal });
  text(s, 'нефтяных\nстволов', 914, 302, 290, 82, 30, { color: C.teal });
  text(s, 'Как K-means группирует фонд по дебиту нефти,\nобводнённости и газовому фактору', 64, 445, 1115, 96, 34);
  text(s, d.source.name, 64, 585, 1120, 44, 26, { color: C.muted });
  footer(s, 'Тема 9. Кластеризация данных');
}

function dataSlide(d) {
  const s = slide('Открытые данные BOEM за 2023 год',
    `Используем официальный годовой файл OGOR-A. В нём ${d.counts.raw_rows} записей ` +
    `по ${d.counts.raw_wells} API. API здесь означает реестровый идентификатор ствола. ` +
    `Исходная строка относится к отчётному месяцу и completion, ` +
    `то есть эксплуатационному объекту внутри ствола. Поэтому количество строк нельзя ` +
    `считать количеством независимых скважин.\n\n` +
    `Отбор: ${d.filters.map(f => `${f.label}: из ${f.before} осталось ${f.after}`).join('; ')}. ` +
    `Активные записи должны иметь нефтедобывающий статус. Требование одного активного ` +
    `completion и отсутствия пересекающихся записей предотвращает двойной счёт дней. ` +
    `В итоге изучаем относительно регулярно работающий нефтяной фонд. Новые, ` +
    `простаивающие и смешанные нефтегазовые объекты представлены неполно.\n\n` +
    `Источник: ${d.source.url}\nАрхив: ${d.source.download_url}\n` +
    `Дата получения: ${d.source.accessed_at}. Условия: ${d.source.license_or_terms}`);
  text(s, 'Месячная отчётность по добыче нефти, газа и воды', 64, 126, 1152, 44, 28);
  text(s, nfmt(d.counts.raw_rows), 64, 207, 300, 82, 64, { color: C.teal, bold: true });
  text(s, 'исходных записей', 64, 291, 300, 40, 26);
  text(s, nfmt(d.counts.raw_wells), 64, 384, 300, 82, 64, { color: C.teal, bold: true });
  text(s, 'стволов до отбора', 64, 468, 300, 40, 26);
  const filters = [
    d.filters.find(f => f.label.includes('код O')),
    d.filters.find(f => f.label.includes('12 месяцев')),
    d.filters.find(f => f.label.includes('активный completion')),
    d.filters.find(f => f.label.includes('9 месяцев')),
    d.filters.at(-1),
  ];
  table(s, [['После фильтра', 'Стволов'], ...filters.map(f => [f.label, nfmt(f.after)])],
    { x: 405, y: 192, w: 811, h: 390, widths: [604, 207], size: 26 });
  text(s, 'Один объект модели = один 12-значный API', 405, 602, 810, 38, 27, { bold: true });
  footer(s, 'API учитывает боковой ствол. Отбор ограничивает область применения выводов.');
}

function selectionSlide(d) {
  const s = slide('Выбор числа групп',
    `Проверяем k от 2 до 8. Для каждого значения обучаем K-means с 20 инициализациями. ` +
    `Оптимальная инерция не возрастает с увеличением k. Найденные локальные решения ` +
    `не обязаны быть монотонными. Сама по себе инерция не выбирает k. ` +
    `Основной критерий здесь: максимальный средний силуэт. Для объекта ` +
    `a означает среднее расстояние до остальных объектов своей группы, b означает ` +
    `минимальное среднее расстояние до другой группы. Силуэт равен (b − a) / max(a, b). ` +
    `Он лежит между −1 и 1. Значения около нуля означают пограничное положение.\n\n` +
    `На этой выборке максимум при k=${d.selected_k}, силуэт ${nfmt(d.silhouette, 3)}. ` +
    `Это сравнение вариантов разбиения в выбранных признаках. Оно не проверяет ` +
    `гипотезу, что естественные группы вообще существуют.\n\n` +
    'Источник метода: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html');
  const cats = d.k_scores.map(v => String(v.k));
  chart(s, 'line', { position: { left: 62, top: 173, width: 714, height: 420 },
    title: 'K-means, 20 инициализаций для каждого k',
    categories: cats, lineOptions: { smooth: false }, series: [{ name: 'Средний силуэт', smooth: false, values: d.k_scores.map(v => chartNumber(v.silhouette)),
      line: { fill: C.teal, width: 3 }, marker: { symbol: 'circle', size: 7, fill: C.teal } }],
    hasLegend: false, xAxis: axis('Число групп k', { majorGridlines: null }),
    yAxis: axis('Средний силуэт', { min: Math.min(0, ...d.k_scores.map(v => v.silhouette)), numberFormatCode: '0.00' }),
  });
  text(s, `k = ${d.selected_k}`, 860, 183, 330, 104, 74, { bold: true, color: C.teal });
  text(s, `${nfmt(d.silhouette, 3)}`, 861, 302, 330, 80, 56, { bold: true });
  text(s, 'максимальный\nсредний силуэт', 864, 387, 330, 80, 28);
  text(s, 'Номера групп условны.\nОни не оценивают\nкачество скважины.', 864, 510, 330, 100, 25, { color: C.muted });
  footer(s, 'Силуэт измеряет разделение в пространстве признаков; эталонных классов нет.');
}

function pcaSlide(d) {
  const variance = d.pca.explained_variance_ratio;
  const total = variance.reduce((a, b) => a + b, 0);
  const s = slide('Расположение стволов по трём признакам',
    `Каждая точка соответствует одному стволу. Для изображения три подготовленных ` +
    `признака спроецированы на две главные компоненты. Эти компоненты объясняют ` +
    `${nfmt(total * 100, 1)}% общей дисперсии. Сам K-means обучался по всем трём ` +
    `признакам после log1p и стандартизации, а не по координатам этого рисунка. ` +
    `Границы и расстояния в двумерной проекции могут отличаться от трёхмерных. ` +
    `Доля стволов с отрицательным индивидуальным силуэтом составляет ` +
    `${nfmt(d.negative_silhouette_fraction * 100, 1)}%. У них принадлежность к группе ` +
    `неоднозначна с точки зрения использованной метрики.\n\n` +
    'PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html');
  chart(s, 'scatter', { position: { left: 60, top: 150, width: 890, height: 475 },
    title: 'Проекция PCA',
    series: d.clusters.map((c, i) => {
      const points = d.pca.points.filter(p => p.cluster === c.cluster);
      return { name: `Группа ${c.cluster}`, xValues: points.map(p => chartNumber(p.pc1)), values: points.map(p => chartNumber(p.pc2)),
        fill: palette[i], line: { fill: 'none', width: 0 },
        marker: { symbol: ['circle', 'diamond', 'triangle', 'square'][i % 4], size: 4, fill: palette[i] } };
    }), scatterOptions: { style: 'marker' }, hasLegend: true,
    xAxis: axis(`PC1 (${nfmt(variance[0] * 100, 1)}%)`, { numberFormatCode: '0.0' }),
    yAxis: axis(`PC2 (${nfmt(variance[1] * 100, 1)}%)`, { numberFormatCode: '0.0' }),
  });
  text(s, `${nfmt(total * 100, 1)}%`, 985, 192, 230, 90, 56, { color: C.teal, bold: true });
  text(s, 'дисперсии\nна двух осях', 985, 291, 230, 85, 26);
  text(s, 'Одна точка\nсоответствует\nодному стволу', 985, 452, 230, 112, 26, { color: C.muted });
  footer(s, 'PCA служит иллюстрацией. Кластеризация использует все три признака.');
}

function profilesSlide(d) {
  const s = slide('Чем отличаются группы',
    `В таблице приведены медианы индивидуальных годовых характеристик стволов. ` +
    `Медиана помогает описать типичный профиль при скошенных распределениях. ` +
    `Она не является центром K-means: центры рассчитываются как средние в преобразованном ` +
    `и стандартизованном пространстве.\n\n` +
    d.clusters.map(c => `Группа ${c.cluster}: ${c.n_wells} ${wellWord(c.n_wells)}, медианный дебит ` +
      `${nfmt(c.oil_rate_bpd, 0)} баррелей в сутки работы, обводнённость ` +
      `${nfmt(c.water_cut_pct, 1)}%, газовый фактор ${nfmt(c.gor_scf_bbl, 0)} стандартных кубических футов на баррель.`).join(' ') +
    `\n\nКаждый ствол имеет одинаковый вес. Медианы группы нельзя трактовать как ` +
    `отношения суммарных объёмов всей группы. Номер группы упорядочен по среднему ` +
    `дебиту, поэтому номер сам по себе физического смысла не имеет.`);
  text(s, 'Медианы индивидуальных годовых характеристик', 64, 128, 1152, 45, 28);
  const values = [['Группа', 'Стволов', 'Нефть,\nбарр./сут работы', 'Вода, %', 'GOR,\nст. фут³/барр.'],
    ...d.clusters.map(c => [`${c.cluster}`, nfmt(c.n_wells), nfmt(c.oil_rate_bpd), nfmt(c.water_cut_pct, 1), nfmt(c.gor_scf_bbl)])];
  const h = Math.min(380, Math.max(260, values.length * 78));
  const t = table(s, values, { y: 196, h, widths: [150, 170, 325, 210, 297], size: 26 });
  for (let i = 1; i < values.length; i++) {
    t.getCell(i, 0).text.style = { fontFamily: font, fontSize: 30, color: palette[i - 1], bold: true };
  }
  if (d.clusters.length === 2) {
    const ordered = [...d.clusters].sort((a, b) => a.oil_rate_bpd - b.oil_rate_bpd);
    const [low, high] = ordered;
    text(s, `Медианный дебит группы ${high.cluster} в ${nfmt(high.oil_rate_bpd / low.oil_rate_bpd, 1)} раза выше.\n` +
      `Обводнённость ниже на ${nfmt(low.water_cut_pct - high.water_cut_pct, 1)} процентного пункта.`,
      64, Math.max(520, 214 + h), 1150, 86, 28);
  } else {
    text(s, 'Различия описывают эксплуатационный профиль.\nПричину высокой воды или высокого GOR нужно проверять отдельно.',
      64, Math.max(520, 214 + h), 1150, 86, 28);
  }
  footer(s, 'Медианы в физических единицах не совпадают с центрами K-means.');
}

function examplesSlide(d) {
  const s = slide('Примеры скважин и практическая польза',
    `Для каждой группы показан реальный API, ближайший к её центру в пространстве ` +
    `модели. Мы не выбирали визуально самые удобные примеры.\n\n` +
    d.examples.map(e => `${e.well_id}: группа ${e.cluster}, дебит ${nfmt(e.oil_rate_bpd)} ` +
      `баррелей в сутки работы, вода ${nfmt(e.water_cut_pct, 1)}%, GOR ${nfmt(e.gor_scf_bbl)}.`).join(' ') +
    `\n\nСегментация даёт список похожих объектов для дальнейшего сравнения. ` +
    `Для группы с большей обводнённостью можно изучить динамику воды и нагрузку ` +
    `на систему подготовки. При высоком газовом факторе полезны данные о давлении, ` +
    `газовом режиме и учёте продукции. Условия работы и причины изменения режима ` +
    `из этой таблицы неизвестны. Решение о воздействии на скважину требует дополнительных данных.`);
  table(s, [['API ствола', 'Группа', 'Нефть,\nбарр./сут', 'Вода, %', 'GOR,\nст. фут³/барр.'],
    ...d.examples.map(e => [e.well_id, `${e.cluster}`, nfmt(e.oil_rate_bpd), nfmt(e.water_cut_pct, 1), nfmt(e.gor_scf_bbl)])],
    { y: 166, h: Math.min(325, Math.max(240, (d.examples.length + 1) * 70)), widths: [310, 150, 250, 180, 262], size: 25 });
  text(s, 'Что проверить по каждой группе', 64, 500, 1152, 46, 30, { bold: true, color: C.teal });
  const waterGroup = [...d.clusters].sort((a, b) => b.water_cut_pct - a.water_cut_pct)[0];
  const oilGroup = [...d.clusters].sort((a, b) => b.oil_rate_bpd - a.oil_rate_bpd)[0];
  text(s, `Группа ${waterGroup.cluster}: сравнить динамику воды и нагрузку на подготовку.\n` +
    `Группа ${oilGroup.cluster}: сопоставить режим работы с близкими аналогами.`, 64, 556, 1150, 80, 27);
  footer(s, 'Примеры ближайшие к центрам групп. Выводы служат гипотезами для проверки.');
}

function reliabilitySlide(d) {
  const st = d.stability;
  const half = st.checks.find(c => c.name.includes('полугод'));
  const noLog = st.checks.find(c => c.name === 'Без log1p');
  const noGor = st.checks.find(c => c.name === 'Без газового фактора');
  const robust = st.checks.find(c => c.name === 'RobustScaler');
  const s = slide('Разбиение чувствительно к масштабу признаков',
    'ARI сравнивает два разбиения одних и тех же объектов с поправкой на случайное ' +
    'совпадение. Единица означает полное совпадение, значение около нуля означает ' +
    'уровень случайного совпадения. Без эталонных классов ARI не является точностью.\n\n' +
    `При смене seed ARI от ${nfmt(st.seed_ari_min, 3)} до ${nfmt(st.seed_ari_max, 3)}. ` +
    `Для Ward при том же k ARI ${nfmt(st.ward_ari, 3)}, силуэт ${nfmt(st.ward_silhouette, 3)}. ` +
    `В десяти подвыборках по 80% объектов ARI от ${nfmt(st.subsample_ari_min, 3)} до ` +
    `${nfmt(st.subsample_ari_max, 3)}, медиана ${nfmt(st.subsample_ari_median, 3)}. ` +
    'При каждом таком отборе scaler и модель обучаются заново, сравнение ведётся на общих объектах.\n\n' +
    st.checks.map(c => `${c.name}: ARI ${nfmt(c.ari, 3)}. ${c.note}`).join('\n') +
    '\n\nРазбиение устойчиво к проверенным инициализациям и к исключению части объектов, ' +
    'но сильно меняется при замене StandardScaler на RobustScaler. Число групп в этих ' +
    'проверках фиксировано. Поэтому результат следует интерпретировать вместе с выбранным ' +
    'масштабированием, а не как единственное естественное разбиение фонда.' +
    '\n\nОграничения:\n' + d.caveats.join('\n') +
    '\n\nARI: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html');
  const values = [['Проверка при том же k', 'ARI'],
    ['Три других seed', `${nfmt(st.seed_ari_min, 3)}…${nfmt(st.seed_ari_max, 3)}`],
    ['Ward вместо K-means', nfmt(st.ward_ari, 3)],
    ['80% стволов, медиана 10 повторов', nfmt(st.subsample_ari_median, 3)],
    [robust.name, nfmt(robust.ari, 3)],
    [noLog.name, nfmt(noLog.ari, 3)],
    [noGor.name, nfmt(noGor.ari, 3)],
    [half.name, nfmt(half.ari, 3)]];
  table(s, values, { y: 163, w: 740, h: 432, widths: [548, 192], size: 24 });
  text(s, nfmt(robust.ari, 3), 856, 180, 340, 75, 56, { bold: true, color: C.ochre });
  text(s, 'ARI при замене\nмасштабирования', 856, 269, 340, 80, 27);
  text(s, 'При смене seed\nгруппы совпадают.\n\nRobustScaler заметно\nменяет их состав.', 856, 399, 350, 190, 27);
  footer(s, 'ARI = 1: одинаковое разбиение. Полугодия отражают и динамику, и чувствительность модели.');
}

function sourceSlide(d) {
  const s = slide('Источники и воспроизведение',
    `Основной результат: для ${d.counts.selected_wells} выбранных стволов выделены ` +
    `${d.selected_k} ${groupWord(d.selected_k)}, средний силуэт ${nfmt(d.silhouette, 3)}. Профили описывают ` +
    `совместные различия дебита, обводнённости и газового фактора. Выбор признаков ` +
    `и преобразований влияет на результат, поэтому мы сохраняем проверки чувствительности.\n\n` +
    `Данные BOEM: ${d.source.url}\nАрхив OGOR-A: ${d.source.download_url}\n` +
    'K-means и другие методы: https://scikit-learn.org/stable/modules/clustering.html\n' +
    'Силуэт: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html\n' +
    'ARI: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html\n' +
    `Репозиторий: https://github.com/dmitriyjob/Neftegaz_practice\n\n` +
    `Для воспроизведения после установки зависимостей: ${d.reproduce_command}. ` +
    'Зафиксированный архив данных позволяет выполнить основной анализ без сети. ' +
    'Версии библиотек и SHA256 исходных файлов сохраняются в run_manifest.json.');
  text(s, `${nfmt(d.counts.selected_wells)} стволов, ${d.selected_k} ${groupWord(d.selected_k)}, силуэт ${nfmt(d.silhouette, 3)}`,
    64, 143, 1152, 62, 36, { bold: true, color: C.teal });
  const sources = [
    ['BOEM OGOR-A 2023', 'Исходная месячная отчётность и описание полей', d.source.url],
    ['scikit-learn', 'K-means, Ward, силуэт, ARI и стандартизация', 'https://scikit-learn.org/stable/modules/clustering.html'],
    ['Репозиторий проекта', 'Данные, код, отчёт, тесты и все результаты', 'https://github.com/dmitriyjob/Neftegaz_practice'],
  ];
  sources.forEach((row, i) => {
    const y = 242 + i * 97;
    const h = text(s, row[0], 64, y, 390, 44, 30, { bold: true });
    h.hyperlink = { uri: row[2], isExternal: true };
    text(s, row[1], 465, y + 3, 750, 64, 27);
  });
  text(s, d.reproduce_command, 64, 564, 1152, 54, 34, { bold: true, color: C.teal });
  footer(s, 'После установки requirements.txt. Зафиксированный архив, версии и хеши включены в проект.');
}

for (const key of ['source', 'counts', 'filters', 'clusters', 'examples', 'k_scores', 'pca', 'stability', 'caveats']) requireField(summary, key);
if (summary.profile_statistic !== 'median') throw new Error('This deck expects median cluster profiles.');
if (summary.clusters.length > 5) throw new Error('More than five groups require adapting the profile and example slide layouts.');
if (summary.clusters.reduce((sum, c) => sum + c.n_wells, 0) !== summary.counts.selected_wells)
  throw new Error('Cluster counts do not sum to the reported sample size.');
if (summary.pca.points.length !== summary.counts.selected_wells)
  throw new Error('PCA must contain exactly one point per selected wellbore.');
if (summary.examples.length !== summary.selected_k || summary.examples.some(e => e.representative_role !== 'nearest_centroid'))
  throw new Error('A verified nearest-centroid example is required for each cluster.');
cover(summary);
methodSlide();
dataSlide(summary);
featuresSlide(summary.source.url);
selectionSlide(summary);
pcaSlide(summary);
profilesSlide(summary);
examplesSlide(summary);
reliabilitySlide(summary);
sourceSlide(summary);
await writeOutputs();
