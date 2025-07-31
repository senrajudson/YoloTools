from ray.tune import ExperimentAnalysis
import pandas as pd

exp_path = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\runs\detect\exp_000001"
analysis = ExperimentAnalysis(exp_path)

df = analysis.dataframe()
print(df.head())

df = pd.DataFrame(df)

df.to_csv(r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\training\data.csv", index=False, encoding='utf-8')

print(f"✅ Exportado com sucesso")