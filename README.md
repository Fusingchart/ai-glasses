# AI Glasses 🥽

DIY AR ambient intelligence glasses — real-time info overlay from your docs + the web.

**Core loop:** hear → understand → retrieve → render card

## Hardware
- **Display**: Xreal Air 2 (waveguide AR, USB-C)
- **Mic**: XIAO ESP32-S3 Sense (BLE audio stream)
- **Compute**: Android phone or Pi 5
- **Audio**: Bone conduction transducer
- **Budget**: ~$250–320

## Phases
- **Phase 1** — Core AI Pipeline + Fake Display ← current
- **Phase 2** — Xreal Display Integration
- **Phase 3** — Glasses Hardware
- **Phase 4** — Intelligence Layer

## Phase 1 Goal
Prove the loop works end-to-end. No glasses hardware needed.

Speak *"I've been reading about attention mechanisms"* → card appears with a doc match from your indexed notes within 4 seconds.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python pipeline.py   # start the AI pipeline
python overlay.py    # launch the fake AR overlay
```
