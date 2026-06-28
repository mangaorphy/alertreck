"""Alertreck outdoor enclosure (INMP441 I2S mic) — parametric CadQuery model.

Generates two editable STEP solids to import into Onshape:
    alertreck_base.step   alertreck_lid.step

Run:
    pip install cadquery
    python alertreck_enclosure.py

In Onshape: Insert > Import each .step — they arrive as B-rep solids you can
keep editing (add the Gore vent, adjust the Pi position, etc.).
"""
import cadquery as cq

# ─── parameters (mm) — tweak these ────────────────────────────────────────────
WALL        = 3.0                   # wall / floor thickness
IL, IW, IH  = 120.0, 90.0, 55.0     # internal length / width / height
CLR         = 0.4                   # lid-to-wall fit clearance
LID_LIP     = 8.0                   # depth the lid skirt drops into the box
SKIRT_T     = 2.0                   # skirt wall thickness

BOSS_D      = 9.0                   # corner screw-boss diameter
LID_SCREW   = 3.2                   # M3 clearance in lid
BOSS_PILOT  = 2.7                   # M3 self-tap pilot in boss

PI_DX, PI_DY = 58.0, 49.0           # Raspberry Pi 4 mounting-hole pattern
PI_STAND_D, PI_STAND_H = 6.0, 6.0
PI_PILOT    = 2.3                   # M2.5 self-tap pilot

# INMP441 bottom-port MEMS mic — verify against your board
MIC_INSET   = 22.0                  # port inset from a corner
MIC_PORT_D  = 5.0                   # acoustic through-hole
MEMB_D      = 14.0                  # hydrophobic membrane recess (underside)
MEMB_REC    = 1.0
MIC_HOLES   = 11.0                  # INMP441 mount-hole spacing (measure yours)
MIC_STAND_D, MIC_STAND_H = 4.0, 3.0
MIC_PILOT   = 1.6                   # M2 pilot

GLAND_D     = 12.5                  # PG7 cable gland, on the floor (sheds water)
EAR_W, EAR_D, EAR_T = 18.0, 16.0, 4.0   # tree-mount ears
EAR_HOLE    = 4.2                   # M4 strap/screw

# ─── derived ──────────────────────────────────────────────────────────────────
OL, OW  = IL + 2 * WALL, IW + 2 * WALL
CORNERS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
BX, BY  = OL / 2 - WALL - BOSS_D / 2, OW / 2 - WALL - BOSS_D / 2
MX, MY  = OL / 2 - MIC_INSET, -(OW / 2 - MIC_INSET)


def cyl(d, h, pos):
    return cq.Workplane("XY").circle(d / 2).extrude(h).translate(pos)


def boxp(l, w, h, pos):
    return cq.Workplane("XY").box(l, w, h, centered=(True, True, False)).translate(pos)


# ─── base: open-top shelled box ───────────────────────────────────────────────
base = (cq.Workplane("XY")
        .box(OL, OW, IH + WALL, centered=(True, True, False))
        .faces(">Z").shell(-WALL))

for sx, sy in CORNERS:                                   # lid screw bosses
    base = base.union(cyl(BOSS_D, IH, (sx * BX, sy * BY, WALL)))
    base = base.cut(cyl(BOSS_PILOT, IH + 1, (sx * BX, sy * BY, WALL)))

for sx, sy in CORNERS:                                   # Pi 4 standoffs
    base = base.union(cyl(PI_STAND_D, PI_STAND_H, (sx * PI_DX / 2, sy * PI_DY / 2, WALL)))
    base = base.cut(cyl(PI_PILOT, PI_STAND_H + 1, (sx * PI_DX / 2, sy * PI_DY / 2, WALL)))

# INMP441: downward acoustic port + membrane recess + two M2 mount posts
base = base.cut(cyl(MIC_PORT_D, WALL + 2, (MX, MY, -1)))
base = base.cut(cyl(MEMB_D, MEMB_REC, (MX, MY, 0)))
for sx in (-1, 1):
    base = base.union(cyl(MIC_STAND_D, MIC_STAND_H, (MX + sx * MIC_HOLES / 2, MY, WALL)))
    base = base.cut(cyl(MIC_PILOT, MIC_STAND_H + 1, (MX + sx * MIC_HOLES / 2, MY, WALL)))

for gx in (-14.0, 12.0):                                 # downward cable glands
    base = base.cut(cyl(GLAND_D, WALL + 2, (gx, OW / 2 - 22, -1)))

for ex in (-OL / 4, OL / 4):                             # tree-mount ears
    cy = -(OW / 2 + EAR_D / 2 - 0.6)
    base = base.union(boxp(EAR_W, EAR_D, EAR_T, (ex, cy, 0)))
    base = base.cut(cyl(EAR_HOLE, EAR_T + 2, (ex, cy, -1)))

# ─── lid: plate + skirt ───────────────────────────────────────────────────────
lid = cq.Workplane("XY").box(OL, OW, WALL, centered=(True, True, False))
for sx, sy in CORNERS:
    lid = lid.cut(cyl(LID_SCREW, WALL + 2, (sx * BX, sy * BY, -1)))
so_l, so_w = IL - 2 * CLR, IW - 2 * CLR
skirt = (cq.Workplane("XY").box(so_l, so_w, LID_LIP, centered=(True, True, False))
         .cut(cq.Workplane("XY").box(so_l - 2 * SKIRT_T, so_w - 2 * SKIRT_T, LID_LIP + 2,
                                     centered=(True, True, False)))
         .translate((0, 0, -LID_LIP)))
lid = lid.union(skirt)

# ─── export ───────────────────────────────────────────────────────────────────
cq.exporters.export(base, "alertreck_base.step")
cq.exporters.export(lid,  "alertreck_lid.step")
print("wrote alertreck_base.step and alertreck_lid.step")
