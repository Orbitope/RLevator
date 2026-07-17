using System.IO;
using ContentKit;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.UI;
using UnityEditor;
using UnityEditor.SceneManagement;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// One-click scene and asset setup for the ElevatorRL environment.
    /// Run via  Tools > Elevator RL > Setup Scene.
    /// </summary>
    public static class ElevatorSetup
    {
        const string AssetFolder   = "Assets/ElevatorRL/Config";
        const string PresetFolder  = "Assets/ElevatorRL/Config/Presets";
        const string FontFolder    = "Assets/ElevatorRL/UI/Fonts";
        const string SandboxScene  = "Assets/Scenes/ElevatorSandbox.unity";
        const string ThemePath     = "Packages/com.mwburke.contentkit/ScriptableObjects/CKDefaultTheme.asset";

        /// <summary>
        /// Toggles Play mode via script — there's no built-in menu item for the Play button (it's
        /// toolbar-only) and no dedicated tool for it in the MCP Unity bridge, so this is the
        /// reliable way to start/stop an mlagents-learn training connection remotely.
        /// </summary>
        [MenuItem("Tools/Elevator RL/Toggle Play Mode")]
        static void TogglePlayMode() => EditorApplication.isPlaying = !EditorApplication.isPlaying;

        /// <summary>
        /// Points the ElevatorController agent (created by Setup Scene) at the S scale-ladder
        /// preset instead of the generic default BuildingConfig.asset. Needed for the M1 parity
        /// gate ("PPO matches LOOK on rung S") to be a valid comparison — the default asset is
        /// numElevators=10/randomizeActive=true, a different problem than rung S (8fl/3cars/fixed
        /// fleet). One-off fix, not folded into SetupScene since it's specific to this milestone.
        /// </summary>
        [MenuItem("Tools/Elevator RL/Point Agent At S Preset (training)")]
        static void PointAgentAtSPreset() => PointAgentAtPreset("S");

        [MenuItem("Tools/Elevator RL/Point Agent At M Preset (training)")]
        static void PointAgentAtMPreset() => PointAgentAtPreset("M");

        [MenuItem("Tools/Elevator RL/Point Agent At L Preset (training)")]
        static void PointAgentAtLPreset() => PointAgentAtPreset("L");

        [MenuItem("Tools/Elevator RL/Point Agent At Z Preset (training)")]
        static void PointAgentAtZPreset() => PointAgentAtPreset("Z");

        [MenuItem("Tools/Elevator RL/Point Agent At H Preset (training)")]
        static void PointAgentAtHPreset() => PointAgentAtPreset("H");

        static void PointAgentAtPreset(string presetName)
        {
            var go = GameObject.Find("ElevatorController");
            if (go == null) { Debug.LogError("[ElevatorRL] No 'ElevatorController' GameObject in the open scene — run Setup Scene first."); return; }
            var agent = go.GetComponent<ElevatorControllerAgent>();
            if (agent == null) { Debug.LogError("[ElevatorRL] ElevatorController has no ElevatorControllerAgent component."); return; }

            string path = $"Assets/ElevatorRL/Config/Presets/{presetName}_BuildingConfig.asset";
            var preset = AssetDatabase.LoadAssetAtPath<BuildingConfig>(path);
            if (preset == null) { Debug.LogError($"[ElevatorRL] {presetName}_BuildingConfig preset not found — run Generate Scale Ladder Presets first."); return; }

            agent.buildingConfig = preset;
            ConfigureBrainParameters(go, agent);
            EditorUtility.SetDirty(go);
            EditorSceneManager.MarkSceneDirty(go.scene);
            Debug.Log($"[ElevatorRL] ElevatorController.buildingConfig -> {path} " +
                      $"({preset.numFloors}fl/{preset.numElevators}cars, randomizeActive={preset.randomizeActive})");
        }

        // EXPERIMENT_PLAN.md E5 (observation ablations): swap ElevatorController's
        // observationConfig without touching buildingConfig, then rebake VectorObservationSize
        // (which changes with the obs block toggles) the same way PointAgentAtPreset does for
        // fleet-size changes.
        [MenuItem("Tools/Elevator RL/E5 Obs Ablations/Point Agent At Full-State Obs Config")]
        static void PointAgentAtObsFullState() => PointAgentAtObsConfig("ObservationConfig_FullState");

        [MenuItem("Tools/Elevator RL/E5 Obs Ablations/Point Agent At Realistic Obs Config")]
        static void PointAgentAtObsRealistic() => PointAgentAtObsConfig("ObservationConfig_Realistic");

        [MenuItem("Tools/Elevator RL/E5 Obs Ablations/Point Agent At Realistic+WaitAge Obs Config")]
        static void PointAgentAtObsRealisticPlusWaitAge() => PointAgentAtObsConfig("ObservationConfig_RealisticPlusWaitAge");

        [MenuItem("Tools/Elevator RL/E5 Obs Ablations/Point Agent At Omniscient Obs Config")]
        static void PointAgentAtObsOmniscient() => PointAgentAtObsConfig("ObservationConfig_Omniscient");

        // Baseline observation (the default ObservationConfig used by all E2/E3/E6 headline runs incl.
        // bignet2). Use this to wire the scene back to baseline obs for E12 traffic-pattern runs, which
        // reuse bignet2's recipe (768x4, baseline obs) so traffic pattern is the only variable.
        [MenuItem("Tools/Elevator RL/E5 Obs Ablations/Point Agent At Baseline Obs Config")]
        static void PointAgentAtObsBaseline() => PointAgentAtObsConfig("ObservationConfig");

        // Baseline + pattern one-hot (E12): gives the policy an explicit traffic-regime signal
        // instead of making it infer the regime from indirect hall-call statistics. obsSize = 259
        // (254 baseline + 5 pattern one-hot) on rung M.
        [MenuItem("Tools/Elevator RL/E12 Traffic Patterns/Point Agent At Pattern-Aware Obs Config")]
        static void PointAgentAtObsPatternAware() => PointAgentAtObsConfig("ObservationConfig_PatternAware");

        // E13b: attach/detach the floor-axis conv observation pathway (FloorGridSensorComponent).
        // Adds a (1 x F x 8) grid sensor next to the agent; train with vis_encode_type: match3. Toggle
        // idempotently so re-running is safe. Rebuild the headless trainer after adding/removing.
        [MenuItem("Tools/Elevator RL/E13 Conv/Add Floor-Grid Sensor To Agent")]
        static void AddFloorGridSensor()
        {
            var go = GameObject.Find("ElevatorController");
            if (go == null) { Debug.LogError("[ElevatorRL] No 'ElevatorController' GameObject in the open scene — run Setup Scene first."); return; }
            if (go.GetComponent<FloorGridSensorComponent>() != null) { Debug.Log("[ElevatorRL] FloorGridSensorComponent already present."); return; }
            go.AddComponent<FloorGridSensorComponent>();
            EditorUtility.SetDirty(go);
            EditorSceneManager.MarkSceneDirty(go.scene);
            Debug.Log("[ElevatorRL] Added FloorGridSensorComponent (floor-axis conv pathway). Rebuild the headless trainer.");
        }

        // E13d: the origin x destination (2 x F x F) conv pathway. Both axes are ordered floors, so
        // the conv kernel is meaningful, and it carries destinations (info the flat obs lacks) so it
        // can move the asymptote — unlike E13b's (1 x F x 8) grid. Remove the E13b floor-grid sensor
        // first: vis_encode_type applies to ALL visual obs and an 8-wide grid breaks resnet.
        [MenuItem("Tools/Elevator RL/E13 Conv/Add Floor-OD Sensor To Agent")]
        static void AddFloorODSensor()
        {
            var go = GameObject.Find("ElevatorController");
            if (go == null) { Debug.LogError("[ElevatorRL] No 'ElevatorController' GameObject in the open scene — run Setup Scene first."); return; }
            if (go.GetComponent<FloorGridSensorComponent>() != null)
                Debug.LogWarning("[ElevatorRL] FloorGridSensorComponent (E13b, 8-wide) is still attached — " +
                    "remove it before training with vis_encode_type: resnet (min-res 15).");
            if (go.GetComponent<FloorODSensorComponent>() != null) { Debug.Log("[ElevatorRL] FloorODSensorComponent already present."); return; }
            go.AddComponent<FloorODSensorComponent>();
            EditorUtility.SetDirty(go);
            EditorSceneManager.MarkSceneDirty(go.scene);
            Debug.Log("[ElevatorRL] Added FloorODSensorComponent (2 x F x F origin×destination conv). Rebuild the headless trainer.");
        }

        [MenuItem("Tools/Elevator RL/E13 Conv/Remove Floor-OD Sensor From Agent")]
        static void RemoveFloorODSensor()
        {
            var go = GameObject.Find("ElevatorController");
            if (go == null) { Debug.LogError("[ElevatorRL] No 'ElevatorController' GameObject in the open scene."); return; }
            var comp = go.GetComponent<FloorODSensorComponent>();
            if (comp == null) { Debug.Log("[ElevatorRL] No FloorODSensorComponent to remove."); return; }
            Object.DestroyImmediate(comp);
            EditorUtility.SetDirty(go);
            EditorSceneManager.MarkSceneDirty(go.scene);
            Debug.Log("[ElevatorRL] Removed FloorODSensorComponent. Rebuild the headless trainer.");
        }

        [MenuItem("Tools/Elevator RL/E13 Conv/Remove Floor-Grid Sensor From Agent")]
        static void RemoveFloorGridSensor()
        {
            var go = GameObject.Find("ElevatorController");
            if (go == null) { Debug.LogError("[ElevatorRL] No 'ElevatorController' GameObject in the open scene."); return; }
            var comp = go.GetComponent<FloorGridSensorComponent>();
            if (comp == null) { Debug.Log("[ElevatorRL] No FloorGridSensorComponent to remove."); return; }
            Object.DestroyImmediate(comp);
            EditorUtility.SetDirty(go);
            EditorSceneManager.MarkSceneDirty(go.scene);
            Debug.Log("[ElevatorRL] Removed FloorGridSensorComponent. Rebuild the headless trainer.");
        }

        static void PointAgentAtObsConfig(string assetName)
        {
            var go = GameObject.Find("ElevatorController");
            if (go == null) { Debug.LogError("[ElevatorRL] No 'ElevatorController' GameObject in the open scene — run Setup Scene first."); return; }
            var agent = go.GetComponent<ElevatorControllerAgent>();
            if (agent == null) { Debug.LogError("[ElevatorRL] ElevatorController has no ElevatorControllerAgent component."); return; }

            string path = $"{AssetFolder}/{assetName}.asset";
            var cfg = AssetDatabase.LoadAssetAtPath<ObservationConfig>(path);
            if (cfg == null) { Debug.LogError($"[ElevatorRL] {path} not found."); return; }

            agent.observationConfig = cfg;
            ConfigureBrainParameters(go, agent);
            EditorUtility.SetDirty(go);
            EditorSceneManager.MarkSceneDirty(go.scene);
            Debug.Log($"[ElevatorRL] ElevatorController.observationConfig -> {path}");
        }

        /// <summary>
        /// Bakes BehaviorParameters.BrainParameters.ActionSpec/VectorObservationSize into the
        /// SAVED SCENE at editor time, computed from whatever configs are currently assigned to
        /// the agent. This must be done here, not in ElevatorControllerAgent.Initialize() — Unity
        /// ML-Agents' actuator/policy setup reads BrainParameters.ActionSpec BEFORE the Agent's
        /// user Initialize() override runs, so setting it at runtime is too late once a stale
        /// value has already been serialized (e.g. from an earlier Setup Scene run against a
        /// different-car-count BuildingConfig) — this produced a real
        /// "Action Mask is too large for specified branch" crash in Play mode. Confirmed pattern:
        /// Pushman's PushmanSetup.cs configures ActionSpec the same way, at editor/setup time.
        /// Call this any time buildingConfig, observationConfig, or actionSpace changes.
        /// </summary>
        static void ConfigureBrainParameters(GameObject go, ElevatorControllerAgent agent)
        {
            var bp = go.GetComponent<BehaviorParameters>();
            if (bp == null) { Debug.LogError("[ElevatorRL] No BehaviorParameters on " + go.name); return; }

            // Building's constructor only allocates arrays from config references — safe/cheap to
            // build a throwaway instance purely to compute ObservationSize() at editor time.
            var tempBuilding = new Building(agent.buildingConfig, agent.rewardConfig, agent.observationConfig, agent.trafficConfig, 1);

            int E = agent.buildingConfig.numElevators;
            int branchSize = agent.actionSpace == ActionSpaceMode.TargetFloor ? agent.buildingConfig.numFloors : 6;
            var branches = new int[E];
            for (int i = 0; i < E; i++) branches[i] = branchSize;

            bp.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(branches);
            bp.BrainParameters.VectorObservationSize = tempBuilding.ObservationSize();
            EditorUtility.SetDirty(bp);
            Debug.Log($"[ElevatorRL] Baked BrainParameters: {E} branches x size {branchSize} " +
                      $"({agent.actionSpace}), VectorObservationSize={tempBuilding.ObservationSize()}");
        }

        [MenuItem("Tools/Elevator RL/Setup Scene")]
        static void SetupScene()
        {
            EnsureFolder();

            var building = GetOrCreate<BuildingConfig>("BuildingConfig");
            var reward   = GetOrCreate<RewardConfig>("RewardConfig");
            var obs      = GetOrCreate<ObservationConfig>("ObservationConfig");
            var traffic  = GetOrCreate<TrafficConfig>("TrafficConfig");

            // ---- ElevatorController GameObject ----
            var existing = GameObject.Find("ElevatorController");
            var go = existing != null ? existing : new GameObject("ElevatorController");

            var agent = go.GetOrAddComponent<ElevatorControllerAgent>();
            agent.buildingConfig    = building;
            agent.rewardConfig      = reward;
            agent.observationConfig = obs;
            agent.trafficConfig     = traffic;

            var bp = go.GetOrAddComponent<BehaviorParameters>();
            bp.BehaviorName = "ElevatorController";
            bp.BehaviorType = BehaviorType.HeuristicOnly;

            // ---- ElevatorRenderer child ----
            Transform rendererChild = go.transform.Find("View");
            if (rendererChild == null)
            {
                var viewGO = new GameObject("View");
                viewGO.transform.SetParent(go.transform, false);
                rendererChild = viewGO.transform;
            }
            var renderer = rendererChild.gameObject.GetOrAddComponent<ElevatorRenderer>();
            agent.view = renderer;

            EditorUtility.SetDirty(go);
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(go.scene);

            Debug.Log("[ElevatorRL] Scene setup complete. Press Play with Heuristic Only to run the baseline policy.");
            Selection.activeGameObject = go;
        }

        [MenuItem("Tools/Elevator RL/Reset Config Assets to Defaults")]
        static void ResetConfigs()
        {
            EnsureFolder();
            GetOrCreate<BuildingConfig>("BuildingConfig");
            GetOrCreate<RewardConfig>("RewardConfig");
            GetOrCreate<ObservationConfig>("ObservationConfig");
            GetOrCreate<TrafficConfig>("TrafficConfig");
            Debug.Log("[ElevatorRL] Config assets created/reset at " + AssetFolder);
        }

        static void EnsureFolder()
        {
            if (!AssetDatabase.IsValidFolder(AssetFolder))
                AssetDatabase.CreateFolder("Assets/ElevatorRL", "Config");
        }

        static T GetOrCreate<T>(string name) where T : ScriptableObject
        {
            string path = $"{AssetFolder}/{name}.asset";
            var existing = AssetDatabase.LoadAssetAtPath<T>(path);
            if (existing != null) return existing;

            var asset = ScriptableObject.CreateInstance<T>();
            AssetDatabase.CreateAsset(asset, path);
            AssetDatabase.SaveAssets();
            return asset;
        }

        // ── Scale-ladder presets (EXPERIMENT_PLAN.md §3) ───────────────────────

        /// <summary>
        /// Generates the five scale-ladder BuildingConfig presets (S/M/L/Z/H). Z and H use
        /// NESTED (not partitioned) floor ranges — every car's range starts at floor 0 — because
        /// BuildingConfig.floorRange is a single contiguous [min,max] per car, and virtually all
        /// traffic originates or terminates at the lobby (floor 0): a car whose range excluded it
        /// could never serve lobby-bound/lobby-origin riders, permanently stranding anyone whose
        /// only eligible cars were such a car. Nesting still creates real scarcity — only the
        /// high-rise cars ever reach the top floors — without that failure mode. This mirrors a
        /// common real design (all banks reach the sky lobby; only some continue further up).
        /// </summary>
        [MenuItem("Tools/Elevator RL/Generate Scale Ladder Presets (S-M-L-Z-H)")]
        static void GenerateScaleLadderPresets()
        {
            EnsureFolder(PresetFolder);

            CreatePreset("S_BuildingConfig", floors: 8, cars: 3, capacity: 8, floorRange: null);
            CreatePreset("M_BuildingConfig", floors: 16, cars: 5, capacity: 8, floorRange: null);
            CreatePreset("L_BuildingConfig", floors: 30, cars: 8, capacity: 8, floorRange: null);

            // Z — zoned, fixed fleet: low-rise (2 cars, 0-14), mid-rise (3 cars, 0-22),
            // high-rise (3 cars, 0-29, full range). Weighted toward high-rise: the top-exclusive
            // band (floors 23-29) is only reachable by the high-rise cars and its round trip is
            // long (~90-150s at floorTravelTime=1.6s), so it needs real dedicated capacity — an
            // earlier 3/3/2 split left that band structurally under-capacity (34-68% abandonment
            // even at very low overall intensity, since only 2 cars could ever reach it and they
            // weren't exclusive to it besides).
            var zRange = new Vector2Int[8];
            for (int i = 0; i < 2; i++) zRange[i] = new Vector2Int(0, 14);
            for (int i = 2; i < 5; i++) zRange[i] = new Vector2Int(0, 22);
            for (int i = 5; i < 8; i++) zRange[i] = new Vector2Int(0, 29);
            CreatePreset("Z_BuildingConfig", floors: 30, cars: 8, capacity: 8, floorRange: zRange);

            // H — zoned, FIXED fleet: low-rise (3 cars, 0-15), mid-rise (3 cars, 0-28),
            // high-rise/express (4 cars, 0-39, full range). Same high-rise-weighted rationale as
            // Z, scaled up (top-exclusive band is floors 29-39). Fixed fleet so H sits on the same
            // saturation-calibration footing as S/M/L/Z for E1/E3/E4 (scale + zoning).
            //
            // NOTE: H previously ALSO carried randomizeActive/serviceChangeProbability (variable
            // fleet) — but a randomly-selected out-of-service subset isn't zone-aware, so an
            // unlucky draw could gut a specific zone's capacity entirely, and abandonment doesn't
            // average out gracefully once queues blow past maxWait — this made H uncalibratable
            // at ANY intensity (26% abandonment even near-zero load). Variable fleet + zoning
            // together is a real, separate robustness question (E7), not the same axis as
            // saturation vs. scale/zoning (E1/E3/E4) — split into its own preset below so each
            // experiment tests one thing at a time.
            var hRange = new Vector2Int[10];
            for (int i = 0; i < 3; i++) hRange[i] = new Vector2Int(0, 15);
            for (int i = 3; i < 6; i++) hRange[i] = new Vector2Int(0, 28);
            for (int i = 6; i < 10; i++) hRange[i] = new Vector2Int(0, 39);
            CreatePreset("H_BuildingConfig", floors: 40, cars: 10, capacity: 8, floorRange: hRange);

            // H_VarFleet — E7 ONLY (fleet-size generalization/robustness): identical topology and
            // zoning to H, but with randomizeActive + mid-episode service changes turned on. Run
            // this at a fixed, comfortably-below-saturation intensity (well under H's own
            // calibrated base) — the point is measuring graceful degradation as active fleet size
            // varies, not finding another saturation curve.
            CreatePreset("H_VarFleet_BuildingConfig", floors: 40, cars: 10, capacity: 8, floorRange: hRange,
                randomizeActive: true, minActive: 5, serviceChangeProb: 0.02f);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("[ElevatorRL] Generated scale ladder presets (S/M/L/Z/H) at " + PresetFolder);
        }

        static BuildingConfig CreatePreset(string name, int floors, int cars, int capacity,
            Vector2Int[] floorRange, bool randomizeActive = false, int minActive = 1,
            float serviceChangeProb = 0f)
        {
            string path = $"{PresetFolder}/{name}.asset";
            var cfg = AssetDatabase.LoadAssetAtPath<BuildingConfig>(path);
            bool isNew = cfg == null;
            if (isNew) cfg = ScriptableObject.CreateInstance<BuildingConfig>();

            cfg.numFloors = floors;
            cfg.numElevators = cars;
            cfg.capacity = capacity;
            cfg.floorRange = floorRange;
            cfg.randomizeActive = randomizeActive;
            cfg.minActiveElevators = minActive;
            cfg.serviceChangeProbability = serviceChangeProb;

            if (isNew) AssetDatabase.CreateAsset(cfg, path);
            else EditorUtility.SetDirty(cfg);
            return cfg;
        }

        // ── Sandbox Scene setup ────────────────────────────────────────────────

        [MenuItem("Tools/Elevator RL/Setup Sandbox Scene")]
        static void SetupSandboxScene()
        {
            EnsureFolder();
            EnsureFolder(FontFolder);
            EnsureFolder("Assets/Scenes");

            // 1. Config assets
            var building = GetOrCreate<BuildingConfig>("BuildingConfig");
            var reward   = GetOrCreate<RewardConfig>("RewardConfig");
            var obs      = GetOrCreate<ObservationConfig>("ObservationConfig");
            var traffic  = GetOrCreate<TrafficConfig>("TrafficConfig");

            // 2. TMP Font assets
            var monoFont    = GetOrCreateTMPFont("ElevatorFonts/JetBrainsMono",
                                                  $"{FontFolder}/JetBrainsMono SDF.asset");
            var displayFont = GetOrCreateTMPFont("ElevatorFonts/Rajdhani-SemiBold",
                                                  $"{FontFolder}/Rajdhani SDF.asset");

            // 3. Assign fonts to CKDefaultTheme if found
            var theme = AssetDatabase.LoadAssetAtPath<CKTheme>(ThemePath);
            if (theme != null)
            {
                theme.monoFont    = monoFont;
                theme.displayFont = displayFont;
                theme.bodyFont    = displayFont;
                EditorUtility.SetDirty(theme);
                AssetDatabase.SaveAssets();
                Debug.Log("[ElevatorRL] Assigned TMP fonts to CKDefaultTheme.");
            }
            else
            {
                Debug.LogWarning("[ElevatorRL] CKDefaultTheme.asset not found at: " + ThemePath);
            }

            // 4. Create or open scene
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // 5. UI Camera
            var camGO = new GameObject("UICamera");
            var cam   = camGO.AddComponent<Camera>();
            cam.orthographic  = true;
            cam.clearFlags    = CameraClearFlags.SolidColor;
            cam.backgroundColor = CKColor.Void;
            cam.cullingMask   = ~0; // all layers — canvas GameObjects default to "Default" layer

            var camData = camGO.GetOrAddComponent<UniversalAdditionalCameraData>();
            camData.renderPostProcessing = true;
            camData.antialiasing = AntialiasingMode.None;

            // 6. Post-process Volume (bloom for amber/coral glow)
            // Wire manually: add a Volume component (Global), assign CKDefaultTheme's post-process profile.
            // UNITY_URP_AVAILABLE define must be set for CKTheme.postProcessProfile to be accessible.
            var volGO = new GameObject("PostProcessVolume");
            volGO.AddComponent<Volume>().isGlobal = true;

            // 7. Event System
            var esGO = new GameObject("EventSystem");
            esGO.AddComponent<EventSystem>();
            // Project uses New Input System package — must use InputSystemUIInputModule
            esGO.AddComponent<InputSystemUIInputModule>();

            // 8. Canvas — Screen Space Overlay avoids camera-raycast dependency issues
            var canvasGO = new GameObject("Canvas");
            var canvas   = canvasGO.AddComponent<Canvas>();
            canvas.renderMode   = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 0;

            var scaler = canvasGO.AddComponent<CanvasScaler>();
            scaler.uiScaleMode         = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1280, 800);
            scaler.matchWidthOrHeight  = 0.5f;

            canvasGO.AddComponent<GraphicRaycaster>();

            // 9. ElevatorSandbox
            var sandboxGO = new GameObject("ElevatorSandbox");
            sandboxGO.transform.SetParent(canvasGO.transform, false);

            var sb = sandboxGO.AddComponent<ElevatorSandbox>();
            sb.buildingConfigAsset    = building;
            sb.rewardConfigAsset      = reward;
            sb.observationConfigAsset = obs;
            sb.trafficConfigAsset     = traffic;
            sb.theme       = theme;
            sb.monoFont    = monoFont;
            sb.displayFont = displayFont;
            sb.uiRoot      = canvasGO.GetComponent<RectTransform>();

            // stretch to fill canvas
            var sbRt = sandboxGO.GetComponent<RectTransform>();
            if (sbRt == null) sbRt = sandboxGO.AddComponent<RectTransform>();
            sbRt.anchorMin = Vector2.zero;
            sbRt.anchorMax = Vector2.one;
            sbRt.offsetMin = sbRt.offsetMax = Vector2.zero;

            // 10. Save scene
            EditorSceneManager.SaveScene(scene, SandboxScene);
            AssetDatabase.Refresh();

            Debug.Log("[ElevatorRL] Sandbox scene created at " + SandboxScene + ". Press Play to run.");
            Selection.activeGameObject = sandboxGO;
        }

        [MenuItem("Tools/Elevator RL/Generate TMP Fonts Only")]
        static void GenerateFontsOnly()
        {
            EnsureFolder(FontFolder);
            var mono    = GetOrCreateTMPFont("ElevatorFonts/JetBrainsMono",
                                              $"{FontFolder}/JetBrainsMono SDF.asset");
            var display = GetOrCreateTMPFont("ElevatorFonts/Rajdhani-SemiBold",
                                              $"{FontFolder}/Rajdhani SDF.asset");
            Debug.Log($"[ElevatorRL] TMP fonts: mono={mono != null}, display={display != null}");
        }

        static TMP_FontAsset GetOrCreateTMPFont(string resourcePath, string savePath)
        {
            var existing = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(savePath);
            if (existing != null) return existing;

            // TMP_FontAsset.CreateFontAsset requires TMP Essential Resources to be imported.
            // TMP_Settings.instance is null if they haven't been imported yet.
            if (TMP_Settings.instance == null)
            {
                Debug.Log("[ElevatorRL] Importing TMP Essential Resources...");
                TMP_PackageResourceImporter.ImportResources(true, false, false);
                AssetDatabase.Refresh();
            }

            var ttf = Resources.Load<Font>(resourcePath);
            if (ttf == null)
            {
                Debug.LogWarning($"[ElevatorRL] TTF not found at Resources/{resourcePath}");
                return null;
            }

            var fa = TMP_FontAsset.CreateFontAsset(ttf);
            if (fa == null) return null;

            AssetDatabase.CreateAsset(fa, savePath);

            // Embed the atlas texture(s) as sub-assets so they survive domain reloads.
            if (fa.atlasTextures != null)
            {
                foreach (var tex in fa.atlasTextures)
                {
                    if (tex != null)
                    {
                        tex.name = fa.name + " Atlas";
                        AssetDatabase.AddObjectToAsset(tex, fa);
                    }
                }
            }

            // Also embed the atlas population texture if it exists separately
            if (fa.material != null)
                AssetDatabase.AddObjectToAsset(fa.material, fa);

            AssetDatabase.SaveAssets();
            AssetDatabase.ImportAsset(savePath);
            return fa;
        }

        static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            string parent = Path.GetDirectoryName(path).Replace('\\', '/');
            string folder = Path.GetFileName(path);
            AssetDatabase.CreateFolder(parent, folder);
        }
    }

    static class GameObjectExtensions
    {
        public static T GetOrAddComponent<T>(this GameObject go) where T : Component
        {
            var c = go.GetComponent<T>();
            return c != null ? c : go.AddComponent<T>();
        }
    }
}
