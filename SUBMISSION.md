# CardioLens Devpost text

Solo project by Aman Tebriwal.

## Tagline

Color a schematic heart with CAD, LAD, LCX, and RCA probabilities from the Z-Alizadeh chart.

## Inspiration

Cath reports are tables. Students still learn the arteries as three names on a flat diagram. I wanted the probability to sit on the vessel it belongs to, with the chart features that pushed the number.

## What it does

CardioLens reads the clinical, ECG, lab, and echo columns from the UCI extension of the Z-Alizadeh Sani dataset (303 patients). It estimates overall CAD and significant stenosis for the LAD, LCX, and RCA. LAD, LCX, RCA, and Cath never enter the feature matrix.

A random forest, chosen by stratified 5-fold ROC-AUC, is the live model. Pooled out-of-fold ROC-AUC is 0.926 for CAD, 0.854 for LAD, 0.733 for LCX, and 0.726 for RCA. Accuracy on CAD is 0.865 against a majority baseline of 0.713. LCX and RCA are weaker, and the report says so.

The UI paints each probability on a labeled schematic heart. You rotate, zoom, and click a vessel. SHAP values, in probability units, show which inputs moved that estimate. A banner states this is teaching and decision support, not a diagnosis and not a cath report.

## How we built it

Python for the table. scikit-learn random forest after a bake-off against logistic regression, histogram gradient boosting, and XGBoost. SHAP TreeExplainer with probability output. FastAPI serves the four forests. Next.js, React, and TypeScript draw the form and a React Three Fiber scene. The mesh is generated in the repo from the algebraic heart surface, not downloaded. Vessel paths are snapped to that surface and named LAD, LCX, and RCA.

## Challenges

Keeping Cath and the three vessel labels out of the inputs. The UCI note says CAD means any vessel is stenotic, but row 93 disagrees, so the models follow the file. n is 303, so LCX and RCA AUC stays in the low 0.70s. SHAP had to reconstruct predict_proba or the training script refuses the model. The heart had to read as a heart on integrated graphics, so the scene stays under 30k triangles with no postprocessing.

## Accomplishments

One encoder shared by training and the API. Out-of-fold metrics checked again by pytest against the saved prediction file. Six real sample rows, including single-vessel patterns. Clicking a vessel switches both the color focus and the SHAP panel. Docker runs the API and the web app on one port.

## What I learned

On this table, typical chest pain, age, wall motion, and ejection fraction dominate the CAD explanation. A pretty accuracy on CAD is easy to overread: 216 of 303 labels are already CAD. Vessel-level recall at a fixed 0.50 threshold is the honest hard part.

## What's next

An external cohort. A threshold chosen for sensitivity, not left at 0.50. A real CT mesh if a clinician will review the correspondence. Calibration plotted for the user, not only in the repo.

## Built With

Python, pandas, scikit-learn, SHAP, XGBoost, matplotlib, FastAPI, Next.js, React, TypeScript, React Three Fiber, Three.js, Tailwind CSS

## AI assistance

Cursor's coding agent wrote most of the training script, API, interface, mesh script, tests, and this text. The metrics were computed by running the training script on the UCI file. They were not filled in by hand.
