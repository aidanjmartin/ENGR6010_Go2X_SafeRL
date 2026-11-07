"""Standing test of contact settings: 4 s of PD on the default pose, the base speed, foot hop and drift.

    python scripts/mjx/check_contacts.py <unitree_mujoco>
"""
import os, sys
import mujoco, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from go2x.mjx import go2_model as G  # noqa: E402
from go2x.sim2sim import DEFAULT_POS  # noqa: E402
MD=sys.argv[1]
def run(m, label):
    d = mujoco.MjData(m); ix = G.indices(m)
    d.qpos[:] = G.home_qpos(m, G.standing_height(m)); mujoco.mj_forward(m, d)
    sp, hop = [], []
    feet=[m.geom(f).id for f in G.FEET]
    for t in range(800):
        tau = np.clip(25*(DEFAULT_POS - d.qpos[ix["qadr"]]) - 0.5*d.qvel[ix["vadr"]], -ix["limit"], ix["limit"])
        d.ctrl[ix["act"]] = tau; mujoco.mj_step(m, d)
        if t > 200: sp.append(np.linalg.norm(d.qvel[:2])); hop.append(max(d.geom_xpos[feet,2]-0.022))
    print(f"{label:40s} mean speed {np.mean(sp):.3f} m/s, max foot height {max(hop)*100:.1f} cm, drift {np.linalg.norm(d.qpos[:2]):.3f} m")
run(G.eval_model(MD), "eval (go2.xml)")
for condim in (3, 6):
  for it, ls in ((4, 8), (100, 50)):
    for cone in (0, 1):
        spec = G.training_spec(MD)
        for b in spec.bodies:
            for g in b.geoms:
                if g.name in G.FEET: g.condim = condim
        spec.option.iterations, spec.option.ls_iterations = it, ls
        spec.option.cone = cone
        if cone == 0: spec.option.impratio = 1
        run(spec.compile(), f"condim {condim} it {it} cone {'elliptic' if cone else 'pyramidal'}")
