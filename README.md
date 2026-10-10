# PCA og eigenfaces – MA-447

Kode og rapport til emnet AI Mathematics ved Universitetet i Agder.
Oppgaven tar utgangspunkt i Matthew Turk og Alex Pentlands artikkel
*Eigenfaces for Recognition* (1991).

Rapporten utleder PCA som et egenverdiproblem og forklarer projeksjon,
rekonstruksjonsfeil og beregning med en mindre Gram-matrise. Et eksempel med
fire punkter viser utregningen. Python-forsøket bruker håndskrevne sifre fra
scikit-learn og sammenligner klassifisering med og uten PCA.

Forsøket bruker ikke ansiktsdata og gjenskaper ikke artikkelens målinger av
ansiktsgjenkjenning.

## Rapport og filer

- [Les rapporten](MA447_PCA_Report.pdf).
- [Python-skript](verify_experiment.py) med norske kommentarer.
- [LaTeX-kilde](main.tex) og figurer som kan åpnes i Overleaf.
- [Resultater](verified-results.json), [fordeling av observasjoner](split-indices.json)
  og [testprediksjoner](test-predictions.csv).

## Kjør forsøket

Det dokumenterte forsøket brukte Python 3.12.14. Opprett gjerne et virtuelt
miljø før du installerer pakkene.

```bash
python -m pip install -r requirements.txt
python verify_experiment.py
```

Datasettet følger med scikit-learn. Skriptet skriver resultatfiler og tre
PDF-figurer til samme mappe som skriptet, og erstatter tidligere genererte
filer med de samme navnene. Figurtekstene er på engelsk, som rapporten.

## Resultat fra forsøket

Datasettet har 1 797 bilder med 64 pikselverdier hver. Med tilfeldig frø 42
fordeles de på 1 077 treningsbilder, 360 valideringsbilder og 360 testbilder.
PCA og klassegjennomsnittene tilpasses først på treningsbildene. Valideringen
velger 30 komponenter; ved lik nøyaktighet velges det minste antallet.
Deretter tilpasses modellen på de 1 437 utviklingsbildene før testen.

| Modell | Koordinater | Riktige på testsettet |
| --- | ---: | ---: |
| PCA og nærmeste klassegjennomsnitt | 30 | 316 av 360 (87,78 %) |
| Nærmeste klassegjennomsnitt med alle piksler | 64 | 316 av 360 (87,78 %) |

De 30 komponentene beholder 95,94 % av variansen i utviklingsdataene.
Resultatet gjelder denne fordelingen av data og denne klassifikatoren.

Skriptets kontroller sammenligner firepunktseksemplet med den eksakte
utregningen, PCA med SVD, og Gram-matrisen med kovariansmatrisen. De sjekker
også rekonstruksjonsfeilen og at bruk av alle koordinater gir de samme
prediksjonene som modellen uten PCA. Små flyttallsforskjeller kan forekomme
på andre plattformer.

## Bygg rapporten

Last prosjektfilene inn i Overleaf, sett `main.tex` som hovedfil og bruk
pdfLaTeX. Rapporten bruker UiA-logoen i `images/`. Logoen tilhører
Universitetet i Agder og er hentet fra rapportmalen.

## Artikkel

M. Turk og A. Pentland, «Eigenfaces for Recognition», *Journal of Cognitive
Neuroscience*, 3(1), 71–86, 1991.
[DOI: 10.1162/jocn.1991.3.1.71](https://doi.org/10.1162/jocn.1991.3.1.71).
