import json, base64, pathlib
m = json.load(open("reports/metrics.json"))
img_b64 = base64.b64encode(open("reports/confusion_matrix.png","rb").read()).decode("utf-8")

html = f"""
<h1>ML Experiment Report (Iris)</h1>
<p><b>Accuracy:</b> {m['accuracy']:.3f}</p>
<h2>Confusion Matrix</h2>
<img src="data:image/png;base64,{img_b64}" />
"""

pathlib.Path("reports/public").mkdir(parents=True, exist_ok=True)
open("reports/public/index.html","w").write(html)
print("Report generated at reports/public/index.html")