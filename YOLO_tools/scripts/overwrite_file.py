import re
import os
import ultralytics

def patch_metrics():
    # Encontra o caminho dinâmico de onde o ultralytics foi instalado no container
    lib_dir = os.path.dirname(ultralytics.__file__)
    metrics_path = os.path.join(lib_dir, 'utils', 'metrics.py')

    print(f"Buscando arquivo em: {metrics_path}")

    with open(metrics_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # REGEX ROBUSTO:
    # 1. Procura exatamente 'w = ' (podendo ter espaços)
    # 2. Procura qualquer lista de colchetes: \[.*?\]
    # 3. EXIGE que o comentário oficial do YOLO esteja na mesma linha para evitar falsos positivos
    pattern = r"(w\s*=\s*)\[.*?\](\s*# weights for \[P, R, mAP@0\.5, mAP@0\.5:0\.95\])"
    
    # Substitui preservando o 'w = ' inicial (\g<1>) e o comentário final (\g<2>)
    replacement = r"\g<1>[0.0, 0.2, 0.8, 0.0]\g<2>"

    new_content, count = re.subn(pattern, replacement, content)

    if count > 0:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ Sucesso! Modificamos os pesos do fitness para [0.0, 0.2, 0.8, 0.0].")
    else:
        print("❌ Aviso: Padrão não encontrado. A versão do Ultralytics mudou?")

if __name__ == "__main__":
    patch_metrics()