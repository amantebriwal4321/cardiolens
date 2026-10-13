# CardioLens demo script

Aim for 3 to 5 minutes. Record the browser at http://127.0.0.1:3921 with the API already running. Speak in English, or add English subtitles. Unlisted YouTube is fine. Private is not.

## 0:00 to 0:25. The problem

Show the disclaimer banner first. Read it once: this is teaching and decision support, not a diagnosis and not a cath report.

Say: the chart can give a probability, but the probability still has to land on LAD, LCX, or RCA. CardioLens does that for the UCI Z-Alizadeh extension, 303 patients.

## 0:25 to 1:10. What the model is allowed to see

Scroll the form just enough to show demographics, one history toggle, and the echo block. Say the inputs are age, history, symptoms, exam, ECG, labs, and echo.

Say clearly: LAD, LCX, RCA, and Cath are the labels. They are not inputs. The API returns 400 if you send them.

## 1:10 to 2:10. A quiet chart and a heavy chart

Click sample A, "No stenotic vessel". Point at the four probabilities. On the unrounded row they sit under 0.50, and the arteries stay toward teal. Mention the source labels: Cath Normal, all three vessels Normal. Also mention the chart is not healthy: this row has severe valve disease coded in the file. The model is not a valve diagnosis.

Click sample B, "Three-vessel pattern". Wait for the update. Probabilities move up: CAD about 73%, LAD about 67%, LCX about 58%, RCA about 67%. The front vessel and the right vessel go warmer. Say these rows were in the final fit, so this is an illustration. The fair numbers are in the cross-validation card.

## 2:10 to 3:20. The heart

Drag to rotate. Scroll to zoom. Click Front view to reset.

Click the LAD tube, then LCX, then RCA. Each click should switch the right-hand panel to that vessel's SHAP list. Read one line, for example typical chest pain pushing toward CAD, and one line pushing the other way.

Say the mesh is a schematic algebraic heart with labeled centerlines, not this patient's CT. Color means probability of stenosis at or above 50% narrowing, not a measured percent stenosis.

## 3:20 to 4:20. The numbers

Open the cross-validation card. Read pooled ROC-AUC: CAD 0.926, LAD 0.854, LCX 0.733, RCA 0.726. Mention CAD accuracy 0.865 versus a majority baseline of 0.713. Say LCX and RCA recall at 0.50 is only about one half, so those vessels are the weak spot.

If there is time, open one physiological group and show a value next to its SHAP bar.

## 4:20 to 4:45. Close

Say the stack in one sentence: scikit-learn forest, SHAP, FastAPI, Next.js, React Three Fiber. Point at the README for the Docker command. Repeat that a clinician still has to read the angiogram.
