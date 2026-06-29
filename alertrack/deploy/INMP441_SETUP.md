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
