# INMP441 I2S Microphone — Setup for Alertreck (Raspberry Pi 4)

Replaces the USB microphone with the **INMP441 I2S MEMS** mic. The INMP441 is a
*digital* mic on the Pi's **I2S** bus — it does **not** show up as a USB audio
device. After this setup it becomes the ALSA **default**, resampled to the model's
44.1 kHz mono, so the daemon needs only `MIC_DEVICE_INDEX = None` (already set).

> ⚠️ **The INMP441 is a 3.3 V device. Never wire VDD to 5 V — it will damage the mic.**

---

## 1. Wiring (INMP441 → Raspberry Pi 40-pin header)

| INMP441 pin | Function        | Pi signal (BCM) | Pi physical pin |
|-------------|-----------------|-----------------|-----------------|
| VDD         | Power 3.3 V     | 3V3             | **1**           |
| GND         | Ground          | GND             | **6**           |
| SCK         | Bit clock (BCLK)| GPIO18 / PCM_CLK| **12**          |
| WS          | Word select (LR)| GPIO19 / PCM_FS | **35**          |
| SD          | Serial data out | GPIO20 / PCM_DIN| **38**          |
| L/R         | Channel select  | GND → LEFT      | **9** (or 14)   |

- **L/R → GND** puts the mic on the **left** channel (what `asound.conf` reads).
  Tie it to 3.3 V instead only if you change the routing to the right channel.
- Keep the SCK/WS/SD wires short (< ~15 cm) — I2S is a clocked digital bus.

```
        INMP441                 Raspberry Pi 4 (physical pins)
       ┌─────────┐
   VDD ┤o        │── 3V3  ........ pin 1
   GND ┤o        │── GND  ........ pin 6   (and L/R → pin 9 GND)
   SD  ┤o        │── GPIO20 ...... pin 38
   WS  ┤o        │── GPIO19 ...... pin 35
   SCK ┤o        │── GPIO18 ...... pin 12
   L/R ┤o        │── GND  ........ pin 9
       └─────────┘
```

---

## 1b. Combined wiring — INMP441 (I2S) **and** SIM808 (UART) together

The mic and the SIM808 do **not** share any signal pin. The INMP441 sits on the
Pi's **I2S bus** (GPIO18/19/20); the SIM808 sits on the **UART** (GPIO14/15). The
only thing they share is **ground**, which is required, not a conflict — every GND
pin on the Pi is the same electrical rail, so each device just uses its own.

| Device | Wire | Pi signal | Pi physical pin |
|--------|------|-----------|-----------------|
| INMP441 | VDD  | 3V3 (**3.3 V only**) | **1**  |
| INMP441 | GND  | GND                  | **6**  |
| INMP441 | L/R  | GND → left channel   | **9**  |
| INMP441 | SCK  | GPIO18 / I2S BCLK    | **12** |
| INMP441 | WS   | GPIO19 / I2S FS      | **35** |
| INMP441 | SD   | GPIO20 / I2S DIN     | **38** |
| SIM808  | RXD  | GPIO14 / TXD         | **8**  |
| SIM808  | TXD  | GPIO15 / RXD         | **10** |
| SIM808  | GND  | GND (common ground)  | **20** |
| SIM808  | VCC  | **external 5 V ≥2 A supply — NOT the Pi's 5 V** | — |

Signal pins used: mic → **12, 35, 38**; SIM808 → **8, 10**. They never overlap.
(The mic's L/R sits on GND pin 9; the SIM808 uses a *different* GND pin — 20 — so
there is no contention even though both need ground.)

```
Raspberry Pi 4 · 40-pin header (pin 1 = top-left)

 pin 1  [3V3 ]──VDD INMP441          pin 2  [5V ]
 pin 5  [    ]                       pin 6  [GND]──GND INMP441
 pin 7  [    ]                       pin 8  [TXD]──RXD SIM808
 pin 9  [GND ]──L/R INMP441          pin 10 [RXD]──TXD SIM808
 pin 11 [    ]                       pin 12 [I2S CLK ]──SCK INMP441
 pin 13 [    ]                       pin 14 [GND     ]
 pin 19 [    ]                       pin 20 [GND     ]──GND SIM808
   ...                               pin 35 [I2S FS  ]──WS  INMP441
                                     pin 38 [I2S DATA]──SD  INMP441
```

**Three things that actually matter (not the pin numbers):**

1. **SIM808 VCC → external 5 V ≥2 A supply, never the Pi's 5 V pin.** The GSM radio
   pulls ~2 A bursts when transmitting; powering it from the Pi browns out and
   reboots the Pi mid-alert. Wire only the SIM808's TX, RX, and GND to the Pi, and
   connect the external supply's ground to a Pi GND (pin 20 above = common ground).
2. **INMP441 is 3.3 V only** — VDD to pin 1 (3V3), never a 5 V pin.
3. **UART is crossed, not straight:** SIM808 **TXD → Pi pin 10 (RXD)**, SIM808
   **RXD → Pi pin 8 (TXD)**. If SMS/GPS returns nothing, swap these two first — it is
   the most common cause.

**Boot config for both buses at once** (`/boot/firmware/config.txt`) — all four lines
coexist because I2S and UART are separate peripherals:

```
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
enable_uart=1
dtoverlay=disable-bt
```

Then free the UART for the SIM808: `sudo systemctl disable hciuart` and reboot. The
SIM808 comes up on `/dev/ttyAMA0` at 9600 baud, matching `SIM808_PORT` in `config.py`.

---

## 2. Enable I2S + the soundcard overlay

Edit the boot config (Bookworm: `/boot/firmware/config.txt`; older OS: `/boot/config.txt`):

```bash
sudo nano /boot/firmware/config.txt
```

Add (or uncomment) these lines, then save:

```
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

The `googlevoicehat-soundcard` overlay is the standard, well-tested way to expose
an INMP441 as an ALSA capture card. Reboot:

```bash
sudo reboot
```

After reboot, confirm the card exists:

```bash
arecord -l
# expect a card named "sndrpigooglevoi" (snd_rpi_googlevoicehat_soundcard)
```

---

## 3. Make it the default mic (resample + downmix)

Install the provided ALSA config so any app gets the mic at the rate/channels it asks for:

```bash
sudo cp alertrack/deploy/asound.conf /etc/asound.conf
sudo alsactl kill quit 2>/dev/null; sudo alsa force-reload 2>/dev/null || true
```

(If `arecord -l` showed a card name other than `sndrpigooglevoi`, edit
`/etc/asound.conf` to match before copying.)

---

## 4. Test the mic

Raw ALSA capture (clap/speak during the 5 s, then play it back):

```bash
arecord -D default -f S16_LE -r 44100 -c 1 -d 5 /tmp/mic.wav
aplay /tmp/mic.wav
```

Then the Alertreck-level test (captures at 44.1 kHz mono like the daemon):

```bash
python3 alertrack/deploy/test_i2s_mic.py
```

A working mic shows `peak`/`rms` clearly rising when you make noise and near-zero
on silence. **The absolute level is low** (INMP441 is a quiet mic) — that's normal;
the daemon's EBU-R128 loudness normalisation scales it before the model sees it.

---

## 5. Run the daemon

No code change needed beyond `MIC_DEVICE_INDEX = None` (already set in `config.py`).

```bash
python3 -m alertrack.main
```

If you prefer not to override the system default, set in `config.py`
`MIC_DEVICE_INDEX = "alertreck_mic"` and skip the `pcm.!default` part of
`asound.conf` (keep only the `pcm.alertreck_mic` block).

---

## 6. Notes & troubleshooting

- **No analog hum.** Unlike the USB mic, I2S is digital, so there's no 50/60 Hz
  mains hum. The `HPF_ENABLED` high-pass in `config.py` is now mostly redundant —
  leave it on (harmless, removes wind rumble) or set `HPF_ENABLED = False` and
  compare field detections.
- **Silent capture (peak ≈ 0):** re-check L/R → GND, the SD/WS/SCK pins, and that
  `arecord -l` lists the card (overlay loaded). Loose SD wire is the usual cause.
- **`arecord` works but Python is silent:** PortAudio is opening the wrong device.
  Set `MIC_DEVICE_INDEX = "googlevoicehat"` (name substring) in `config.py`.
- **Card name differs / overlay not found:** some images use a different overlay;
  alternatively follow Adafruit's "I2S MEMS microphone" custom-overlay guide, then
  point `asound.conf`/`MIC_DEVICE_INDEX` at that card.
- **Sample rate:** the mic runs 48 kHz natively; `plug` resamples to the 44.1 kHz
  the models were trained on. Do **not** change `SAMPLE_RATE` in `config.py`.

---

## 7. Field enclosure (stacked Pi 4 + SIM808)

A weatherproof, FDM-printable enclosure that stacks the Pi 4 (lower) and SIM808 EVB
(upper) is generated parametrically:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
    cad/alertreck_stack_enclosure.py
```

Outputs to `cad/`: `alertreck_stack_body.{stl,step}`, `alertreck_stack_lid.{stl,step}`,
`alertreck_stack_sim_door.stl`. Internal cavity 100×70×65 mm, 3 mm walls / 4 mm floor.

- **Standoffs:** Pi on 4× M2.5 posts **7 mm** tall (58×49 pattern); SIM808 on 4×
  corner posts to **Z=37 mm** (10 mm above the Pi top), kept just outside the Pi
  footprint so they don't foul the board.
- **Antennas:** two 8 mm SMA bulkhead holes (GSM + GPS) on the **+X wall** (Face 2,
  upper zone, beside the Pi USB/Ethernet); plus an 8 mm GPS port in the **lid centre**
  for a sky-facing patch antenna.
- **SIM access:** sliding door on the **+Y wall** (Face 4, `sim_card_door.stl`),
  openable from outside without removing the lid.
- **Wiring note:** the INMP441 (I2S) and SIM808 (UART) wire to the GPIO header
  internally per §1b — the box needs **no** USB port for the mic. The Pi USB-A
  cutouts remain for service access only.
- **PETG print settings:** 0.2 mm layer, 4 perimeters, 30 % gyroid infill, body
  printed open-face-up. See the header of `cad/alertreck_stack_enclosure.py` for the
  full face map and the design notes.
