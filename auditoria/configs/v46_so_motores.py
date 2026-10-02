import sys, os, importlib.util
pasta = sys.argv[1]; sys.argv = ["x"]
sp = importlib.util.spec_from_file_location("vf", os.path.join(pasta, "validar_funil.py")); VF = importlib.util.module_from_spec(sp); sp.loader.exec_module(VF)
try:
    M = VF.motores(); print("motores() OK:", {k: K["motor_info"] for k, (B, K) in M.items()})
except SystemExit as e:
    print("SystemExit:", e); sys.exit(3)
