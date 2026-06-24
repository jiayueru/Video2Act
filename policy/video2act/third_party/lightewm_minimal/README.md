# LightEWM Minimal Wan2.2 VAE Backend

This directory contains the minimal LightEWM source needed by Video2Act data
precaching. It is used only by `policy/video2act/scripts/precache_wan22_latents.py`
to load `Wan2.2_VAE.pth` and encode RGB frames into Wan2.2 VAE latents.

Source reference:

- Upstream local source: `/share/jiayueru/Video2Act_code/Lightewm`
- Upstream revision used when this copy was created:
  `fb7057f4b5b854c8b6fd11abe96dacc0facd538b`

Copied/adapted files:

- `wan_video_vae.py`: copied from
  `lightewm/model/wan/wan_video_vae.py`
- `loader.py`: local minimal loader for `WanVideoVAE38`

This is intentionally not a full LightEWM vendor copy. If you need a newer or
full LightEWM checkout, update this directory from upstream or adapt the
precache script to use the external project.
