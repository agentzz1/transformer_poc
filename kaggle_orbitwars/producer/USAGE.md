# orbit-lite

Self-contained single-game Orbit Wars agent (torch + stdlib only).

In a Kaggle notebook (add this dataset, then):

    import sys
    sys.path.insert(0, "/kaggle/input/orbit-lite")
    import orbit_lite                      # the package
    # full agent:
    sys.path.insert(0, "/kaggle/input/orbit-lite")  # so main.py finds orbit_lite
    import importlib.util
    spec = importlib.util.spec_from_file_location("lite_main", "/kaggle/input/orbit-lite/main.py")
    m = importlib.util.module_from_spec(spec); sys.modules["lite_main"] = m; spec.loader.exec_module(m)
    moves = m.agent(obs)                   # [[from_planet_id, angle, ships], ...]
