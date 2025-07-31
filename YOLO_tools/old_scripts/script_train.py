import signal, subprocess, time, os

PYTHON = r"C:/Users/AUTOU4/AppData/Local/pypoetry/Cache/virtualenvs/venv-project-yolo-JkQOy1Sx-py3.12/Scripts/python.exe"
SCRIPT = "tune_train.py"
DURACAO = 1 * 60 * 60  # 30 minutos
N_CICLOS = 60

for i in range(1, N_CICLOS + 1):
    print(f"\n=== Ciclo {i}/{N_CICLOS}")
    proc = subprocess.Popen([PYTHON, SCRIPT], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    print(f"PID={proc.pid}. Rodando por {DURACAO}s…")

    try:
        time.sleep(DURACAO)
        print("Tempo atingido — enviando CTRL_C_EVENT.")
        os.kill(proc.pid, signal.CTRL_C_EVENT)
    except KeyboardInterrupt:
        print("Interrupção manual — enviando CTRL_C_EVENT.")
        os.kill(proc.pid, signal.CTRL_C_EVENT)
    finally:
        proc.wait()
        print(f"Ciclo {i} finalizado com código {proc.returncode}")

print("\n✅ Todos os ciclos completos — use resume=True para continuar.")
