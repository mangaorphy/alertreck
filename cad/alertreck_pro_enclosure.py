"""Alertreck PRO enclosure — Raspberry Pi 4 + SIM808, rounded two-piece case.

A modern, rounded, vented two-piece case (base + screw-fastened lid) that exposes
ALL Raspberry Pi 4 ports on the correct edges, plus SIM808 antenna/SIM openings.

  ⚠️  NOT weatherproof (open ports + top vents). This is the bench/enclosure build.
      For outdoor use see alertreck_field_enclosure.py (sealed).

Coordinate convention: corner origin. Inner cavity spans x∈[WALL, WALL+IL],
y∈[WALL, WALL+IW], floor top at z=WALL. The Pi sits in the −Y/−X corner with its
USB-C corner at (PX, PY):
    • −Y long wall  → USB-C, micro-HDMI 0/1, 3.5 mm audio
    • +X short wall → Ethernet + 2×USB3 + 2×USB2
    • −X short wall → micro-SD access
    • +Y wall       → SIM808 GPS + GSM SMA bulkheads + SIM tray slot
All port offsets are VARIABLES — verify against the official RPi 4 mechanical
drawing and tweak. Real Pi 4 PCB = 85×56 mm, holes 58×49 (corner inset 3.5).

Exports:  alertreck_pro_base.step   alertreck_pro_lid.step
Run:      pip install --only-binary=:all: cadquery && python alertreck_pro_enclosure.py
Print:    PLA/PETG, 0.4 mm nozzle, ~96×124 mm footprint (fits 220×220 bed).
"""
import cadquery as cq

# ─── global shell ─────────────────────────────────────────────────────────────
WALL      = 2.8                     # wall / floor thickness (2.5–3 mm)
CLR       = 2.5                     # clearance board↔wall
CORNER_R  = 6.0                     # outer vertical-edge radius (the "rounded" look)
RIM_CH    = 1.0                     # top-rim chamfer for soft transition
H_BASE    = 24.0                    # internal clear height of the base
STANDOFF  = 3.0                     # board sits this high off the floor
PCB_T     = 1.4

# ─── Raspberry Pi 4 (official) ────────────────────────────────────────────────
PI_L, PI_W   = 85.0, 56.0
PI_DX, PI_DY = 58.0, 49.0           # hole pattern
PI_HOLE_IN   = 3.5                  # corner inset of first hole
PX, PY       = WALL + CLR, WALL + CLR          # USB-C corner of the PCB
PI_STAND_D, PI_PILOT = 6.0, 2.3                 # M2.5 self-tap

# board-top datum and connector z-centres (from PCB top)
ZB        = WALL + STANDOFF + PCB_T
Z_USBC    = ZB + 1.6
Z_HDMI    = ZB + 1.5
Z_AUDIO   = ZB + 3.0
Z_USB     = ZB + 7.8
Z_ETH     = ZB + 6.75
Z_SD      = ZB - 0.5

# ── Pi ports.  −Y long edge: (x_off_from_USBC_corner, width_x, height_z, z_centre)
PORTS_LONG = [
    ('usb_c',  7.7, 11.0, 5.0, Z_USBC),
    ('hdmi0', 26.0,  8.0, 4.5, Z_HDMI),
    ('hdmi1', 39.5,  8.0, 4.5, Z_HDMI),
    ('audio', 54.0,  8.0, 8.0, Z_AUDIO),
]
# ── +X short edge: (y_off_from_USBC_corner, width_y, height_z, z_centre)
PORTS_SHORT = [
    ('usb2',  9.00, 15.0, 17.0, Z_USB),
    ('usb3', 27.00, 15.0, 17.0, Z_USB),
    ('eth',  45.75, 17.5, 15.0, Z_ETH),
]
SD_Y_OFF = 28.0                     # micro-SD on −X edge

# ─── SIM808 — MEASURE THESE ───────────────────────────────────────────────────
SIM_L, SIM_W   = 70.0, 51.0
SIM_DX, SIM_DY = 64.0, 45.0         # MEASURE mount-hole pattern
GAP_BOARDS     = 6.0                # gap between Pi and SIM808
SX             = WALL + CLR                      # SIM −X corner
SY             = PY + PI_W + GAP_BOARDS          # SIM seated above the Pi
SIM_STAND_D, SIM_PILOT = 6.0, 2.3
SMA_D    = 6.5                      # GPS / GSM SMA bulkhead
Z_SMA    = 14.0
SIM_TRAY = (12.0, 3.0)             # SIM-card tray slot (w_x, h_z) on +Y wall

# ─── derived box size ─────────────────────────────────────────────────────────
IL  = (PX - WALL) + PI_L + 2.5                   # +X wall hugs Pi USB/Eth edge
IW  = (SY - WALL) + SIM_W + CLR                  # +Y wall hugs SIM antenna edge
OL, OW, OH = IL + 2 * WALL, IW + 2 * WALL, WALL + H_BASE

# ─── fastening / mounting ─────────────────────────────────────────────────────
BOSS_D, BOSS_PILOT, LID_SCREW = 8.5, 2.7, 3.2
CB = WALL + CORNER_R + BOSS_D / 2 - 1            # boss centre inset from corner
BOSSES = [(CB, CB), (OL - CB, CB), (CB, OW - CB), (OL - CB, OW - CB)]
FOOT_D, FOOT_REC = 11.0, 0.8                     # rubber-foot recesses (underside)
LID_LIP, SKIRT_T, FIT = 7.0, 2.0, 0.4

# INMP441 mic (optional) — downward port in the gap between boards
MIC_X, MIC_Y = WALL + IL - 14, (PY + PI_W + SY) / 2
MIC_PORT_D, MEMB_D, MEMB_REC = 5.0, 14.0, 1.0


# ─── helpers (absolute corner-origin coordinates) ─────────────────────────────
def box_at(cx, cy, cz, l, w, h):
    return cq.Workplane("XY").box(l, w, h).translate((cx, cy, cz))


def zcyl(d, h, x, y, z0):
    return cq.Workplane("XY").circle(d / 2).extrude(h).translate((x, y, z0))


def ycyl(d, x, z, y0, length):                  # axis +Y (pierce a Y wall)
    return (cq.Workplane("XY").circle(d / 2).extrude(length)
            .rotate((0, 0, 0), (1, 0, 0), -90).translate((x, y0, z)))


def standoffs(solid, x0, y0, dx, dy, inset, d, pilot, h):
    cx, cy = x0 + inset, y0 + inset
    for px in (cx, cx + dx):
        for py in (cy, cy + dy):
            solid = solid.union(zcyl(d, h, px, py, WALL))
            solid = solid.cut(zcyl(pilot, h + 1, px, py, WALL))
    return solid


# ─── base: rounded, shelled box ───────────────────────────────────────────────
outer = cq.Workplane("XY").box(OL, OW, OH).translate((OL / 2, OW / 2, OH / 2))
try:
    outer = outer.edges("|Z").fillet(CORNER_R)
except Exception as e:
    print("corner fillet skipped:", e)
base = outer.faces(">Z").shell(-WALL)

# corner screw bosses + reinforcement
for bx, by in BOSSES:
    base = base.union(zcyl(BOSS_D, H_BASE, bx, by, WALL))
    base = base.cut(zcyl(BOSS_PILOT, H_BASE + 1, bx, by, WALL))

# board standoffs
base = standoffs(base, PX, PY, PI_DX, PI_DY, PI_HOLE_IN, PI_STAND_D, PI_PILOT, STANDOFF)
base = standoffs(base, SX, SY, SIM_DX, SIM_DY, (SIM_W - SIM_DY) / 2,
                 SIM_STAND_D, SIM_PILOT, STANDOFF)

# Pi ports — −Y long wall
for name, xo, w, h, zc in PORTS_LONG:
    base = base.cut(box_at(PX + xo, WALL / 2, zc, w, WALL + 6, h))
# Pi ports — +X short wall
for name, yo, w, h, zc in PORTS_SHORT:
    base = base.cut(box_at(OL - WALL / 2, PY + yo, zc, WALL + 6, w, h))
# micro-SD — −X short wall
base = base.cut(box_at(WALL / 2, PY + SD_Y_OFF, Z_SD, WALL + 6, 14.0, 3.5))

# SIM808 — GPS + GSM SMA bulkheads + SIM tray on +Y wall
sim_cx = SX + SIM_L / 2
base = base.cut(ycyl(SMA_D, sim_cx - 16, Z_SMA, OW - WALL - 3, WALL + 6))
base = base.cut(ycyl(SMA_D, sim_cx + 16, Z_SMA, OW - WALL - 3, WALL + 6))
base = base.cut(box_at(sim_cx, OW - WALL / 2, ZB + SIM_TRAY[1], SIM_TRAY[0], WALL + 6, SIM_TRAY[1]))

# INMP441 mic — downward acoustic port + membrane recess (floor)
base = base.cut(zcyl(MIC_PORT_D, WALL + 2, MIC_X, MIC_Y, -1))
base = base.cut(zcyl(MEMB_D, MEMB_REC, MIC_X, MIC_Y, 0))

# rubber-foot recesses (underside)
for fx, fy in [(CB, CB), (OL - CB, CB), (CB, OW - CB), (OL - CB, OW - CB)]:
    base = base.cut(zcyl(FOOT_D, FOOT_REC, fx, fy, 0))

# soften the top rim
try:
    base = base.edges(">Z").chamfer(RIM_CH)
except Exception as e:
    print("rim chamfer skipped:", e)


# ─── lid: rounded plate + skirt + vents + light pipes ─────────────────────────
lid = cq.Workplane("XY").box(OL, OW, WALL).translate((OL / 2, OW / 2, WALL / 2))
try:
    lid = lid.edges("|Z").fillet(CORNER_R)
except Exception as e:
    print("lid fillet skipped:", e)

# inner skirt that drops into the base
si_l, si_w = IL - 2 * FIT, IW - 2 * FIT
skirt = (cq.Workplane("XY").box(si_l, si_w, LID_LIP)
         .faces(">Z").shell(-SKIRT_T)
         .translate((WALL + IL / 2, WALL + IW / 2, WALL / 2 - LID_LIP / 2 - WALL / 2)))
lid = lid.union(skirt.translate((0, 0, 0)))

# screw holes
for bx, by in BOSSES:
    lid = lid.cut(zcyl(LID_SCREW, WALL + 2, bx, by, -1))

# ventilation slot array over the Pi CPU / RAM region
vent_cx, vent_cy = PX + 42, PY + 28
for i in range(-2, 3):
    for j in range(-3, 4):
        sx_, sy_ = vent_cx + i * 9.0, vent_cy + j * 5.0
        if abs(i) + abs(j) <= 5:                 # diamond-ish decorative cluster
            slot = (cq.Workplane("XY").slot2D(14.0, 3.0, 0).extrude(WALL + 2)
                    .translate((sx_, sy_, -1)))
            lid = lid.cut(slot)

# light pipes for PWR/ACT LEDs (Pi LEDs are by the USB-C/SD corner)
for lx in (PX + 2.0, PX + 6.5):
    lid = lid.cut(zcyl(3.0, WALL + 2, lx, PY + 2.5, -1))

# ─── export ───────────────────────────────────────────────────────────────────
cq.exporters.export(base, "alertreck_pro_base.step")
cq.exporters.export(lid,  "alertreck_pro_lid.step")

bb = base.val().BoundingBox()
print(f"internal  : {IL:.0f} x {IW:.0f} x {H_BASE:.0f} mm")
print(f"external  : {bb.xlen:.0f} x {bb.ylen:.0f} x {bb.zlen:.0f} mm  (fits 220x220 bed)")
print(f"base valid: {base.val().isValid()}   lid valid: {lid.val().isValid()}")
print("wrote alertreck_pro_base.step and alertreck_pro_lid.step")
