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
        const string FontFolder    = "Assets/ElevatorRL/UI/Fonts";
        const string SandboxScene  = "Assets/Scenes/ElevatorSandbox.unity";
        const string ThemePath     = "Packages/com.mwburke.contentkit/ScriptableObjects/CKDefaultTheme.asset";

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
