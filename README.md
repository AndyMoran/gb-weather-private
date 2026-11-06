# GB weather: private working repository

The weather-processing code for Brief 4, the weather study that follows [gb-foundation-model-tests](https://github.com/AndyMoran/gb-foundation-model-tests). Andy Moran (Heaviside Analytics).

**This repository is private by design** (Brief 4, Section 10, option A). The weather-processing code is the author's own method, which he may use in paid work. At the weather study's freeze it is hashed, and the hashes are published in the public repository, so the code cannot change unseen. It is published with the results note, on the same day. Everything a reader needs to check the study during it (the plan, the daily national weather inputs with their hashes, the forecasts and the results) is public from the start.

| Folder | Contents |
| --- | --- |
| `docs/` | The brief (drafts until agreed) |
| `feasibility/` | Checks of ECMWF's open-data archive, ERA5, and each model's weather inputs |
| `weather/` | The weather processing itself, frozen and hashed before any weather forecast is made |

Data are never committed: downloads go to `~/gb-weather-data/`. Credentials stay in `~/.cdsapirc`.

Weather data: "Contains modified Copernicus/ECMWF data" (ECMWF open data, CC-BY-4.0; ERA5, Copernicus licence).
