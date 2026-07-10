# Elevator RL Dashboard — Unity Implementation Plan (ContentKit / uGUI + TMP)

> For the **research framing, baseline explanation, and experiment matrix** (training the RL
> controller and comparing it to LOOK across building scales/zoning), see
> [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md). This document covers only the visualization.


Faithful Unity recreation of the **"Elevator RL Control Sandbox"** dashboard (the
Claude-designed ContentKit dashboard) built on the user's own **ContentKit** Unity
package (`com.mwburke.contentkit`). Heuristic-driven sandbox: it owns its own
`Building`, steps it on the speed timer, and dispatches with
`ElevatorHeuristics.CollectiveLook`. No ML-Agents dependency on the sandbox path;
an `.onnx` model can be slotted later.

**Framework decision: uGUI + TextMeshPro** (NOT UI Toolkit) — because ContentKit is
uGUI/TMP based (`CKColor`, `CKTheme`, `CKHUD`) and ships Cinemachine + Recorder +
URP post-processing for filming. Rendering the dashboard on a **Screen Space –
Camera** canvas through a camera with ContentKit's post-process profile gives us
**bloom**, which recovers the HTML's amber/coral *glow* on hot values and the
door-open highlight — the one thing a plain UI couldn't reproduce.

Sources of truth:
- Layout/behavior: `/Users/mwburke/Downloads/Elevator simulation environment/Elevator RL Sandbox.dc.html` — its `renderVals()` is the exact view-model; mirror it.
- Target look: that folder's `screenshots/running.png`, `loaded.png`.
- ContentKit package: `/Users/mwburke/unity_projects/com.mwburke.contentkit/` (palette `Runtime/CKColor.cs`, theme `Runtime/Theme/CKTheme.cs` + `ScriptableObjects/CKDefaultTheme.asset`, HUD `Runtime/UI/CKHUD.cs`).

---

## 0. Scope / ground rules

- **Do NOT modify the simulation scripts** in `Assets/ElevatorRL/Runtime/` — they are
  complete and correct (`Building`, `Elevator`, `Passenger`, the 4 configs,
  `PassengerArrivals`, `ElevatorHeuristics`, `ElevatorControllerAgent`).
- The old `ElevatorRenderer.cs` (3D primitive cubes) is **not** used by the dashboard.
  Leave the file; don't wire it into the sandbox scene.
- **Reuse ContentKit** — palette via `ContentKit.CKColor.*`, fonts/post-processing via
  `CKDefaultTheme.asset`. Do not hardcode hex; reference `CKColor` (values are identical
  to the HTML's contentkit.css, already verified).
- Build the dashboard with **uGUI (Canvas/RectTransform/Image) + TextMeshPro (TMP_Text)**.

## 1. Package wiring (do this FIRST)

1. Add the local ContentKit package to `Packages/manifest.json` (path is two levels up
   from the project — confirm relative path; contentkit project uses `file:../../com.mwburke.contentkit`,
   so from RLevator it is the same: `file:../../com.mwburke.contentkit`):
   ```json
   "com.mwburke.contentkit": "file:../../com.mwburke.contentkit",
   ```
   Add it near the top of `dependencies` (alongside `com.unity.ml-agents`). Unity will
   **auto-resolve its transitive deps**: `com.unity.cinemachine` 3.1.0,
   `com.unity.recorder` 5.1.0, `com.unity.timeline` 1.8.7 (Timeline already present),
   `com.unity.ugui` (present). TextMeshPro is included via ugui/built-in.
2. Let Unity resolve (focus the Editor; watch `Packages/packages-lock.json` gain
   `com.mwburke.contentkit` + cinemachine + recorder). If TMP "Import TMP Essentials"
   is needed, the editor setup (§8) handles font assets explicitly so this isn't blocking.
3. **asmdef:** create `Assets/ElevatorRL/UI/ElevatorRL.UI.asmdef`:
   ```json
   {
     "name": "ElevatorRL.UI",
     "rootNamespace": "ElevatorRL",
     "references": ["ElevatorRL", "ContentKit.Runtime", "Unity.TextMeshPro", "UnityEngine.UI"],
     "autoReferenced": true
   }
   ```
   And an editor asmdef for the setup additions if needed (or reuse existing
   `ElevatorRL.Editor.asmdef` — add `"ContentKit.Runtime"`, `"ContentKit.Editor"`,
   `"Unity.TextMeshPro"`, `"UnityEngine.UI"` to its `references`).

## 2. Palette & fonts via ContentKit

- **Colors:** `using ContentKit;` then `CKColor.Void`, `CKColor.Surface`, `CKColor.Raised`,
  `CKColor.Border`, `CKColor.TextBright/Primary/Secondary/Muted`,
  `CKColor.Amber/AmberBright/AmberDark`, `CKColor.Steel/SteelBright`,
  `CKColor.Coral/CoralBright`, `CKColor.Sage` (= queue down-bar `#7D9A6A`).
  Optionally read from `CKDefaultTheme.asset` (so a theme swap re-skins the dashboard) —
  expose a `public CKTheme theme;` field, fall back to `CKColor` statics when null.
- **Fonts:** ContentKit's `CKDefaultTheme.asset` has `displayFont`/`bodyFont`/`monoFont`
  **unassigned** (fileID 0). The editor setup (§8) **generates TMP font assets** from the
  `.ttf` files already copied to `Assets/ElevatorRL/UI/Resources/ElevatorFonts/`
  (`JetBrainsMono.ttf`, `Rajdhani-SemiBold.ttf`) and assigns them into `CKDefaultTheme`
  (fixing ContentKit's gap as a bonus). Most text = **mono** (JetBrains); the "ELEVATOR"
  wordmark + 42px reward number = **display** (Rajdhani).
  - **Glyph gotcha:** the ▲ (U+25B2) ▼ (U+25BC) arrows and • dots may be absent from
    JetBrains Mono. Set a **TMP fallback** to the default `LiberationSans SDF` (TMP
    essentials) on the mono font asset, OR render arrows/dots as small `Image`s
    (triangles via a built-in triangle sprite; dots via `Knob`/circle sprite). Prefer
    Image-based dots (5px circles) and Image-based direction triangles for crispness;
    use TMP only for the digits/labels.

## 3. Canvas / camera / post-processing (the "nice" filmic look)

- **UI camera:** a dedicated `Camera` (orthographic, clear color = `CKColor.Void`),
  culling only the UI layer. Add URP post-processing: a `Volume` (global) using
  `CKDefaultTheme.postProcessProfile` (it's already assigned in the asset — guid
  `9bd72ba4...`). This profile is what makes amber/coral **bloom**.
- **Canvas:** `Screen Space - Camera`, render camera = the UI camera, reference
  resolution 1280×800 (matches the HTML `$preview`), `CanvasScaler` = Scale With Screen
  Size, match 0.5. This makes hot values glow and door-open cars rim-light — the HTML's
  `box-shadow`/`text-shadow` effect.
- If bloom proves fiddly, fall back to **Screen Space – Overlay** (crisp, no glow) and
  rely on `AmberBright` color alone. Recorder's *Game View* capture records either mode.
- An `EventSystem` is required for the buttons/sliders/dropdowns to receive input.

## 4. ElevatorSandbox.cs — simulation driver (framework-neutral; unchanged from before)

Place in `Assets/ElevatorRL/UI/ElevatorSandbox.cs` (under the `ElevatorRL.UI` asmdef).
This half is pure C# and identical regardless of UI framework.

### 4.1 Serialized fields (defaults match the screenshots)
```csharp
public CKTheme theme;                       // optional; null => CKColor statics
public BuildingConfig    buildingConfigAsset;
public RewardConfig      rewardConfigAsset;
public ObservationConfig observationConfigAsset;
public TrafficConfig     trafficConfigAsset;

[Header("Sandbox start config")]
public int   startFloors    = 8;
public int   startCars      = 3;
public int   startCapacity  = 8;
public int   speed          = 5;    // 1..20 decisions / second
public float intensity      = 1.0f; // 0..3 traffic multiplier
public TrafficPattern startPattern = TrafficPattern.UpPeak;
public int   seed           = 1;
public bool  showObservation = true;
```

### 4.2 Runtime-cloned configs (never dirty the saved assets)
```csharp
_cfg = Instantiate(buildingConfigAsset); _reward = Instantiate(rewardConfigAsset);
_obs = Instantiate(observationConfigAsset); _traffic = Instantiate(trafficConfigAsset);
_cfg.numFloors = startFloors; _cfg.numElevators = startCars; _cfg.capacity = startCapacity;
_cfg.randomizeActive = false;        // ALL cars in service (HTML has no OOS concept)
_cfg.serviceChangeProbability = 0f; _cfg.minActiveElevators = 1;
_traffic.useDayCycle = false;        // sandbox computes pattern itself (4.5)
_traffic.defaultPattern = startPattern; _traffic.intensity = intensity;
_selectedPattern = startPattern;
BuildSim();
```
If any config asset is unassigned, `ScriptableObject.CreateInstance<...>()` so the
sandbox runs standalone.

```csharp
void BuildSim() {
    _b = new Building(_cfg, _reward, _obs, _traffic, seed);
    _b.Reset();                       // all cars active (randomizeActive=false)
    _decisions = 0; _total = 0; _last = 0;
    _utilSum = new float[_cfg.numElevators]; _utilSteps = 0; _decisionClock = 0;
}
```

### 4.3 Run loop (continuous tick + decision cadence — mirrors the Agent's FixedUpdate)
```csharp
float SimRate => _playing ? speed * _cfg.decisionInterval : 0f; // speed decisions/sec
const float MAX_SUB = 0.05f;

void Update() {
    if (_playing) { Advance(Time.deltaTime * SimRate); RefreshView(); }
}
void Advance(float simDelta) {
    while (simDelta > 1e-6f) {
        float h = Mathf.Min(MAX_SUB, simDelta);
        _b.Tick(h); simDelta -= h; _decisionClock += h;
        if (_decisionClock >= _cfg.decisionInterval) { _decisionClock -= _cfg.decisionInterval; Decide(); }
    }
}
void Decide() {
    _traffic.defaultPattern = ActivePattern();
    var act = ElevatorHeuristics.CollectiveLook(_b);
    for (int i = 0; i < _cfg.numElevators; i++) _b.ApplyAction(i, act[i]);
    _last = _b.CollectReward(); _total += _last;
    for (int i = 0; i < _cfg.numElevators; i++) _utilSum[i] += _b.cars[i].Load;
    _utilSteps++; _decisions++;
}
void StepOnce() {                      // STEP button while paused: advance to next decision
    if (_playing) return;
    int d = _decisions, guard = 0;
    while (_decisions == d && guard++ < 100000) Advance(MAX_SUB);
    RefreshView();
}
```

### 4.4 Cars animate smoothly
Read `Building.cars[i].position` (continuous float, floor units) for vertical placement —
cars glide. Door-open = `state ∈ {DoorsOpening, Dwelling, DoorsClosing}`.

### 4.5 Active pattern + clock (parity with HTML)
```csharp
const int MinutesPerStep = 2;
TrafficPattern ActivePattern() {
    if (!_dayCycle) return _selectedPattern;
    int m = (6*60 + _decisions*MinutesPerStep) % 1440; float h = m/60f;
    if (h>=6 && h<10) return TrafficPattern.UpPeak;
    if (h>=11 && h<14) return TrafficPattern.Lunch;
    if (h>=16 && h<19) return TrafficPattern.DownPeak;
    return TrafficPattern.Midday;
}
string Clock() { int m=(6*60+_decisions*MinutesPerStep)%1440; return $"{m/60:00}:{m%60:00}"; }
```

### 4.6 Controls → effects
| Control | Effect |
|---|---|
| PLAY/PAUSE | toggle `_playing` |
| STEP | `StepOnce()` (dim when `_playing`) |
| RESET | `_playing=false; BuildSim(); RefreshView();` |
| speed 1..20 | `speed = round(v)` |
| traffic 0..3 (.1) | `_traffic.intensity = v` |
| pattern (5) | `_selectedPattern = v` (disabled when `_dayCycle`) |
| DAY CYCLE | `_dayCycle = !_dayCycle` |
| floors 3..16 | `_cfg.numFloors=v; BuildSim(); RebuildDynamicUI();` |
| cars 1..6 | `_cfg.numElevators=v; BuildSim(); RebuildDynamicUI();` |
| capacity 4,6,8,10,12,16 | `_cfg.capacity=v; BuildSim();` |

### 4.7 Private state
```csharp
Building _b; BuildingConfig _cfg; RewardConfig _reward; ObservationConfig _obs; TrafficConfig _traffic;
bool _playing,_dayCycle; TrafficPattern _selectedPattern;
int _decisions,_utilSteps; double _total; float _last,_decisionClock; float[] _utilSum;
```
Color accessors: `Color Amber => theme? theme.amber : CKColor.Amber;` etc. (or just use
`CKColor` directly).

## 5. uGUI construction helpers

Write a small `CKUI` static helper (in the UI asmdef) to keep construction terse:
```csharp
static RectTransform Panel(Transform parent, Color bg);                 // GameObject + Image(bg) + RectTransform
static TMP_Text Label(Transform parent, string text, TMP_FontAsset f, float size, Color c,
                      TextAlignmentOptions align = MidlineLeft, float charSpacing = 0);
static Image Box(Transform parent, Color c);                            // raw Image (bars, dots, dividers)
static void Anchor(RectTransform rt, Vector2 anchorMin, Vector2 anchorMax,
                   Vector2 offsetMin, Vector2 offsetMax);               // absolute placement
static void TopLeft(RectTransform rt, float x, float y, float w, float h); // anchor top-left, px pos+size
static void AddBorder(RectTransform rt, Color c, float w, Edges e);     // optional: thin Image lines for 1px borders
```
- **1px borders/dividers:** uGUI has no per-side border. Emulate with thin child `Image`s
  (a 1px `Border`-colored strip) along the needed edge, or use a 9-sliced outline sprite.
  For the major chrome dividers (header/aside/footer separators) a single 1px `Image` line
  is fine. For floor band top-lines and the gutter right-line, lay 1px `Image`s.
- **Layout:** use `HorizontalLayoutGroup`/`VerticalLayoutGroup` + `LayoutElement` for the
  footer control row and the aside's stacked blocks. Use **manual anchored RectTransforms**
  (TopLeft helper) for the building shaft view (floors/tiles/cars), exactly like the HTML's
  absolute positioning, so cars place by pixel and animate smoothly.
- **Percent-width bars** (wait-fill, utilization, queue): set the fill Image's RectTransform
  anchored left, and set `sizeDelta.x` from `fraction * trackWidth` (cache trackWidth from
  the parent rect), or anchor stretch and drive `anchorMax.x = fraction`.

## 6. Layout spec (mirror the HTML; uGUI terms)

Reference resolution 1280×800. Root full-rect on the canvas, bg `Void`.
```
Root (Image Void, stretch)
├── Header  (top strip h50, Image Surface, 1px Border bottom line)
│   ├── Left:  "ELEVATOR" (display 19, Bold, TextBright, charSpacing ~6) + "RL CONTROL SANDBOX" (mono 11 upper, TextMuted, charSpacing ~18)
│   └── Right: 3 column pairs (gap 26): SIM TIME {Clock()}, PATTERN {patternLabel}, STEP {decisions}
│             label mono10 upper TextMuted; value mono15 (PATTERN value Amber, rest TextPrimary)
├── Main (fills between header & footer)
│   ├── Section (left, fills remaining width, bg Void, padding 16/18)   ← BUILDING VIEW (§6.1)
│   └── Aside   (right, width 336, bg Surface, 1px Border left line, vertical stack, scroll if needed) ← METRICS (§6.2)
└── Footer (bottom strip h64, Image Surface, 1px Border top line, horizontal, padding 18, gap 20) ← CONTROLS (§6.3)
```
Constants (match HTML): `shaftW=46`, `gap=12`, `maxTilesPerTrack=9`,
`fh = clamp(floor((sectionHeight-32)/numFloors),40,86)` (recompute when the section rect
changes; default 62), `shaftsWidth = NE*shaftW+(NE-1)*gap`, `rightPad = shaftsWidth+30`,
`tileH = clamp(floor((fh-16)/2)-1,13,22)`.

### 6.1 Building view (left) — manual anchored placement
Inner content rect: height `NF*fh`, 1px Border bottom line.
**Floor bands**, `f` from `NF-1`→`0`, `top=(NF-1-f)*fh` (y down from top):
- Band rect: x 0, y top, width `sectionWidth-rightPad`, height `fh`; 1px Border top line.
- Left gutter (w54, 1px Border right line): floor label `f==0?"G":f` (mono14 TextPrimary,
  centered) + two call markers row: up ▲ + down ▼:
  - up active if `upQ[f].Count>0`, down active if `downQ[f].Count>0`.
  - active → AmberBright (full alpha); inactive → TextMuted @ 0.3 alpha.
  - **Use small triangle `Image`s** (not glyphs) tinted by state — crisper + bloom-friendly.
- Track area (x54→bandRight, two stacked rows, gap5, padding 0/10):
  - **Up row** then **Down row** (each height tileH, clip overflow):
    - first `maxTilesPerTrack` waiting passengers → tile: w18 h`tileH`, bg `Raised`,
      1px border `urgent?Coral:Border`, radius2; center label `dest==0?"G":dest` (mono9,
      `urgent?CoralBright:TextSecondary`); bottom-left fill `Image` h2 w `min(1,wait/maxWait)*18`
      px, color `urgent?Coral:Amber`. `urgent = waitTime>=0.8*_cfg.maxWait`.
    - overflow → `+N` (mono9 TextMuted), `N=count-9`.
**Shafts** (anchored top-right of inner content, x from `sectionWidth-8-shaftsWidth`):
- per elevator i: shaft rect w`shaftW` h`NF*fh`, bg `Surface`, 1px Border, radius2.
  Optional: NF thin Border lines at each `k*fh` for the floor grid.
  - id "E{i}" top-center (mono9 TextMuted, charSpacing 12).
  - **Car** rect: x3..(shaftW-3), `y=(NF-1-position)*fh+4`, h`fh-8`, bg `Raised`,
    1px border `doorOpen?AmberBright:Amber`, radius3:
    - door strip Image left edge w3 `AmberBright`, shown only when doorOpen (this + bloom
      = the HTML door glow).
    - load `{riders}/{cap}` (mono12 AmberBright).
    - dots: up to `min(riders,18)` 5px circle Images `Amber`, wrap, gap2, maxW36.
    - direction triangle Image `dir>0?▲Amber:▼SteelBright` (8px).

### 6.2 Metrics panel (right aside, w336) — VerticalLayoutGroup, blocks split by 1px Border
1. **Cumulative reward** (pad 18/20/16): "CUMULATIVE REWARD" (mono10 upper TextMuted);
   row: big `_total:F1` (display42 AmberBright) + `(_last>=0?"+":"")+_last:F2 + " / step"`
   (mono13, `_last>=0?AmberBright:SteelBright`).
2. **KPI grid 3×2**: cells delivered=`DeliveredTotal`, thru/step=`_decisions>0?
   (DeliveredTotal/_decisions):F2:"0.00"`, avg wait=`WaitCount>0?(WaitSum/WaitCount):F1:"0.0"`,
   max wait=`Round(MaxWaitObserved)`, rejected=`RejectedTotal`, abandoned=`AbandonedTotal`.
   cell: pad14/16, 1px Border right (cols 0,1) + top (row2); label mono9.5 upper TextMuted;
   value mono20 TextBright. Use GridLayoutGroup (3 cols) or 2 HorizontalLayoutGroups.
3. **Elevator utilization** (pad16/20): title; per car row: "E{i}" (mono11 TextSecondary w22);
   track (h8 bg Void 1px Border radius2) + fill (`avg*100%` Amber); pct (mono11 Amber w34 right).
   `avg=_utilSteps>0?_utilSum[i]/_utilSteps:0`.
4. **Queue length / floor** (pad16/20): header "QUEUE LENGTH / FLOOR" + "▲up ▼dn" (mono9 TextMuted);
   per floor `NF-1`→0: "G"/f (mono10 TextSecondary w18 right) + two stacked 5px bars
   (up `min(100,upQ/maxQueue*100)%` Steel; down Sage `#7D9A6A`) + counts `up/dn` (mono10 TextMuted w30).
5. **Observation · limited** (if `showObservation`, pad16/20/22):
   `car_locations`: `[ {cars[i].Floor joined "  "} ]` (mono12 SteelBright).
   `car_buttons`: per car "E{i} {bits}" bits=`for f: cars[i].WantsFloor(f)?'1':'0'`, color
   Amber if car has riders else TextMuted.
   `hall_buttons [up dn]`: `for f: (upQ>0?1:0)(downQ>0?1:0)` space-joined.

### 6.3 Footer controls (HorizontalLayoutGroup, h64, gap20)
- PLAY/PAUSE Button (TMP label): paused → bg Amber, text Void, Bold; playing → bg Raised,
  1px Amber border, text AmberBright. minW74.
- STEP Button (bg Raised, Border, TextPrimary; alpha .4 when `_playing`).
- RESET Button (same base).
- 1px Border vertical divider h32.
- SPEED: label "SPEED" + `{speed}/s` (Amber) + uGUI `Slider` 1..20 wholeNumbers.
- TRAFFIC: label + `{intensity:F1}×` + `Slider` 0..3.
- PATTERN: label + `TMP_Dropdown` ["Up-peak","Down-peak","Lunch","Midday","Uniform"] →
  enum order; alpha .4 + non-interactable when `_dayCycle`.
- DAY CYCLE Button: `"DAY CYCLE  " + (_dayCycle?"ON":"OFF")`; Amber border/text when on.
- 1px divider.
- FLOORS `TMP_Dropdown` 3..16; CARS 1..6; CAPACITY 4,6,8,10,12,16. (label mono9.5 + dropdown
  bg Raised Border radius3.)

## 7. Refresh strategy

- **BuildUI()** constructs everything once; **RebuildDynamicUI()** rebuilds only the
  floor/shaft subtrees (called on floors/cars dropdown changes). Cache references to all
  mutating elements: header clock/pattern/step; reward number + delta; 6 KPI values; per-car
  util fill width + pct; per-car {car rect y, border color, door strip active, load text, dots
  parent, dir triangle}; per-floor {up/down call triangles, up/down track containers}; queue
  bars; observation lines.
- **RefreshView()** (every `Update` while playing + after step/reset/config change): write
  cached props only. Tile children and dot children are rebuilt each refresh (counts vary) —
  small lists, fine. Use a pooled list or clear+repopulate. Avoid LINQ / per-frame allocs;
  reuse a `StringBuilder` for observation strings and a `char[]`/`string` cache for `n/cap`.
- Car `y` is set every frame from `position` → smooth glide (no tween needed).

## 8. Editor setup — extend `Assets/ElevatorRL/Editor/ElevatorSetup.cs`

Add menu **Tools ▸ Elevator RL ▸ Setup Sandbox Scene** (and a helper to generate fonts):

1. **TMP font assets** (fixes ContentKit's empty fonts):
   ```csharp
   static TMP_FontAsset MakeTMPFont(string resourcePath, string savePath) {
       var ttf = Resources.Load<Font>(resourcePath);             // "ElevatorFonts/JetBrainsMono"
       var fa  = TMP_FontAsset.CreateFontAsset(ttf);             // dynamic SDF
       AssetDatabase.CreateAsset(fa, savePath);                  // "Assets/ElevatorRL/UI/Fonts/JetBrainsMono SDF.asset"
       // add LiberationSans SDF as fallback for ▲▼ glyphs if you use TMP arrows
       return fa;
   }
   ```
   Generate `JetBrainsMono SDF` + `Rajdhani SDF`; assign into `CKDefaultTheme.asset`
   (`monoFont`, `displayFont`) and `EditorUtility.SetDirty` + save. (Load CKDefaultTheme via
   `AssetDatabase.LoadAssetAtPath<CKTheme>` searching by type, or by known package path
   `Packages/com.mwburke.contentkit/ScriptableObjects/CKDefaultTheme.asset`.)
2. Ensure the 4 config assets exist (reuse existing `GetOrCreate<>`).
3. Create scene objects:
   - `EventSystem` (+ `StandaloneInputModule`/`InputSystemUIInputModule` — project uses the
     new Input System, so use `InputSystemUIInputModule`).
   - `UICamera` (orthographic, clearFlags SolidColor = `CKColor.Void`, culling = UI layer).
   - Global `Volume` with `CKDefaultTheme.postProcessProfile` (read from the theme asset).
   - `Canvas` (Screen Space – Camera, renderCamera = UICamera) + `CanvasScaler`
     (ScaleWithScreenSize, 1280×800, match .5) + `GraphicRaycaster`.
   - `ElevatorSandbox` GameObject under the canvas (or a controller object) with the 4 config
     assets + `theme = CKDefaultTheme` assigned; it builds the UI under the Canvas in `OnEnable`/
     `Start` (pass it the Canvas `RectTransform` as the build root — expose
     `public RectTransform uiRoot;` and assign the Canvas rect, or have it find the Canvas).
4. Save scene to `Assets/Scenes/ElevatorSandbox.unity`, leave open. `Selection.activeObject`
   = the sandbox.

> URP note: post-processing requires the camera's `UniversalAdditionalCameraData.renderPostProcessing
> = true` and the active URP renderer to allow it. Set it in the editor setup. If the bloom
> doesn't show, verify the Volume is global + profile has Bloom enabled.

## 9. Verification

1. Add package, let Unity resolve (check `packages-lock.json` for contentkit + cinemachine +
   recorder). Recompile; fix CS errors.
2. **Tools ▸ Elevator RL ▸ Setup Sandbox Scene**, press **Play**.
3. Expect: dark dashboard; 8 floors (G..7), 3 shafts (E0..E2), cars `n/8` with dots + dir
   arrows that **glide** between floors; hall call triangles light (with bloom glow); reward
   number climbs and glows; KPIs/utilization/queue update; door-open cars rim-light; footer
   controls all work (pause/step/reset, speed paces, traffic loads, pattern reshapes arrivals,
   floors/cars/capacity rebuild).
4. Compare to `screenshots/running.png` / `loaded.png` for parity.
5. (Optional) **ContentKit ▸ Recording ▸ Setup Recorder** → Start to film a clip and confirm
   the bloom/look records.

## 10. Polish (after parity)
- Shaft floor-grid lines per shaft.
- Tune the post-process Bloom threshold/intensity so only AmberBright/Coral bloom.
- Optional Cinemachine camera for a slow push-in if filming.
- Later: optional Sentis/`.onnx` policy to replace the heuristic in `Decide()`.

---
### Appendix — why this changed from UI Toolkit to uGUI/TMP
ContentKit (`com.mwburke.contentkit`) is a uGUI + TextMeshPro production toolkit
(`CKColor`, `CKTheme`, `CKHUD`) with Cinemachine + Recorder + URP post-processing for
filming. Building the dashboard in uGUI/TMP on a Screen Space–Camera canvas lets us reuse
the palette/fonts/theme, record cleanly, and recover the HTML's glow via **bloom** — which
plain UI Toolkit could not do. The simulation driver (§4) is identical either way.
```
