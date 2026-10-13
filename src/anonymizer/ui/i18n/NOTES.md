# Translation notes (fr, de, pt, it, zh-Hans)

Terms used the same way in every file (and as in es.json): blur (verb/effect) = flouter/flou, weichzeichnen, desfocar/desfoque,
sfocare/sfocatura, 模糊. "Review/check" = one word per language: vérifier, prüfen, revisar, controllare, 检查. "Not sure" =
incertain, unsicher, incerto, incerto, 不确定 (es: dudoso); the legend, the confirmation title and its body use that one word in each language. "Render anyway" repeats the verb of the Render button that opens the dialog. Item = élément, Element, item, elemento, 条目 (not 项目, which is "project").
Area = zone, Bereich, área, area, 区域. Plate = plaque (d'immatriculation), Kennzeichen, placa, targa, 车牌. Frame = image, Bild,
quadro, fotogramma, 帧. Clip kept as "clip" (de: Clip, pt: clipe, zh: 片段). Render (export step) = générer, erzeugen, gerar, generare, 生成.

Choices that may need a second opinion:
- pt is Brazilian Portuguese (arquivo, quadro, desfoque). European Portuguese would use ficheiro, fotograma, desfocar.
- de uses the formal "Sie"; nav items are nouns (Import, Analyse, Prüfung, Export).
- fr/pt: Intl.PluralRules treats count 0 as "one", so "0 clip" is right, but keys whose `.one` text carries a literal "1"
  (analyze.unchecked.one, export.clip_to_check.one, export.to_check.one, export.summary.one, review.warning.decisions_lost.one) would show "1"
  for count 0. Callers only use them with a count above 0. Counts of 1,000,000+ pick `many` in fr/it/pt, which is absent (falls to the base
  key, or "[key]" when there is none); irrelevant for realistic counts.
- zh-Hans only ever selects "other"; the `.one` keys are kept (same wording) so the key set equals en.json.
- Keys I doubted (sense could shift): options.thoroughness (fr Précision, de Gründlichkeit, pt Minúcia, it Accuratezza, zh 检测精细度);
  face.reason.* / plate.reason.* ("second look" = second passage / zweite Prüfung / segunda verificação / secondo controllo / 二次检查);
  options.gap.hold and settings.gap.hold ("Hold last box"); review.view.split; review.sc.blur ("blur on/off"); review.length ("Length" of the item);
  export.comparison.hint; review.range ("{from} to {to}"); error.analysis.incomplete and analyze.unchecked ("cannot be exported as clean" =
  propre / bereinigt / limpo / pulito / 已清理); report.col.from/to.
