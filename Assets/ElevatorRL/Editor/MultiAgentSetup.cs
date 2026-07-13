using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// One-click scene setup for the E6 multi-agent (per-car, parameter-sharing) architecture —
    /// see EXPERIMENT_PLAN.md E6 and Runtime/BuildingManager.cs / ElevatorCarAgent.cs. Builds a
    /// BuildingManager GameObject plus one ElevatorCarAgent child per car in the target preset, all
    /// sharing Behavior Name "ElevatorCar" so ML-Agents trains one shared policy across every car.
    /// Run via Tools > Elevator RL > E6 Multi-Agent > Setup For &lt;X&gt; Preset, in whatever scene
    /// should host the multi-agent training setup (kept in a separate scene from the single-agent
    /// Training.unity so both architectures stay independently runnable/comparable).
    /// </summary>
    public static class MultiAgentSetup
    {
        const string ConfigDir = "Assets/ElevatorRL/Config";
        const string PresetDir = ConfigDir + "/Presets";

        [MenuItem("Tools/Elevator RL/E6 Multi-Agent/Setup For S Preset (training)")]
        static void SetupS() => Setup("S");

        [MenuItem("Tools/Elevator RL/E6 Multi-Agent/Setup For M Preset (training)")]
        static void SetupM() => Setup("M");

        [MenuItem("Tools/Elevator RL/E6 Multi-Agent/Setup For L Preset (training)")]
        static void SetupL() => Setup("L");

        [MenuItem("Tools/Elevator RL/E6 Multi-Agent/Setup For Z Preset (training)")]
        static void SetupZ() => Setup("Z");

        [MenuItem("Tools/Elevator RL/E6 Multi-Agent/Setup For H Preset (training)")]
        static void SetupH() => Setup("H");

        static void Setup(string presetName)
        {
            var buildingConfig = AssetDatabase.LoadAssetAtPath<BuildingConfig>($"{PresetDir}/{presetName}_BuildingConfig.asset");
            if (buildingConfig == null) { Debug.LogError($"[ElevatorRL] {presetName}_BuildingConfig preset not found — run Generate Scale Ladder Presets first."); return; }

            var reward = AssetDatabase.LoadAssetAtPath<RewardConfig>($"{ConfigDir}/RewardConfig.asset");
            var obs = AssetDatabase.LoadAssetAtPath<ObservationConfig>($"{ConfigDir}/ObservationConfig.asset");
            var traffic = AssetDatabase.LoadAssetAtPath<TrafficConfig>($"{ConfigDir}/TrafficConfig.asset");
            if (reward == null || obs == null || traffic == null)
            {
                Debug.LogError("[ElevatorRL] RewardConfig/ObservationConfig/TrafficConfig not found — " +
                                "run Tools > Elevator RL > Setup Scene once first to generate them.");
                return;
            }

            var root = GameObject.Find("BuildingManager");
            if (root == null) root = new GameObject("BuildingManager");
            var manager = root.GetComponent<BuildingManager>();
            if (manager == null) manager = root.AddComponent<BuildingManager>();
            manager.buildingConfig = buildingConfig;
            manager.rewardConfig = reward;
            manager.observationConfig = obs;
            manager.trafficConfig = traffic;

            // Clear any car agents from a previously-configured preset with a different car count.
            for (int i = root.transform.childCount - 1; i >= 0; i--)
                Object.DestroyImmediate(root.transform.GetChild(i).gameObject);

            // Throwaway Building instance purely to compute CarObservationSize() at editor time —
            // same reasoning as ElevatorSetup.ConfigureBrainParameters (Building's constructor only
            // allocates arrays from config references, safe/cheap to build just for this).
            var tempBuilding = new Building(buildingConfig, reward, obs, traffic, 1);
            int obsSize = tempBuilding.CarObservationSize();
            var branches = new[] { 6 };

            for (int i = 0; i < buildingConfig.numElevators; i++)
            {
                var carGo = new GameObject($"Car_{i}");
                carGo.transform.SetParent(root.transform);

                var bp = carGo.AddComponent<BehaviorParameters>();
                bp.BehaviorName = "ElevatorCar";
                bp.BehaviorType = BehaviorType.Default;
                bp.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(branches);
                bp.BrainParameters.VectorObservationSize = obsSize;

                carGo.AddComponent<ElevatorCarAgent>();
                EditorUtility.SetDirty(carGo);
            }

            EditorUtility.SetDirty(root);
            EditorSceneManager.MarkSceneDirty(root.scene);
            Debug.Log($"[ElevatorRL] E6 multi-agent scene set up for {presetName} " +
                      $"({buildingConfig.numFloors}fl/{buildingConfig.numElevators}cars): " +
                      $"{buildingConfig.numElevators}x ElevatorCarAgent, shared Behavior 'ElevatorCar', " +
                      $"obsSize={obsSize}, action=1 branch x 6.");
        }

        // ------------------------------------------------------------ E6 Architecture B (attention)
        [MenuItem("Tools/Elevator RL/E6 Attention/Setup For S Preset (training)")]
        static void SetupAttentionS() => SetupAttention("S");

        [MenuItem("Tools/Elevator RL/E6 Attention/Setup For M Preset (training)")]
        static void SetupAttentionM() => SetupAttention("M");

        [MenuItem("Tools/Elevator RL/E6 Attention/Setup For L Preset (training)")]
        static void SetupAttentionL() => SetupAttention("L");

        [MenuItem("Tools/Elevator RL/E6 Attention/Setup For Z Preset (training)")]
        static void SetupAttentionZ() => SetupAttention("Z");

        [MenuItem("Tools/Elevator RL/E6 Attention/Setup For H Preset (training)")]
        static void SetupAttentionH() => SetupAttention("H");

        // Single-agent + BufferSensor attention (EXPERIMENT_PLAN.md E6 / plan Architecture B). One
        // ElevatorController GameObject: VectorSensor = global obs (baked VectorObservationSize),
        // BufferSensorComponent = per-car entities (ObservableSize=CarEntitySize, MaxNumObservables=
        // numElevators), E-branch joint MultiDiscrete action. Behavior "ElevatorAttention".
        static void SetupAttention(string presetName)
        {
            var buildingConfig = AssetDatabase.LoadAssetAtPath<BuildingConfig>($"{PresetDir}/{presetName}_BuildingConfig.asset");
            if (buildingConfig == null) { Debug.LogError($"[ElevatorRL] {presetName}_BuildingConfig preset not found — run Generate Scale Ladder Presets first."); return; }

            var reward = AssetDatabase.LoadAssetAtPath<RewardConfig>($"{ConfigDir}/RewardConfig.asset");
            var obs = AssetDatabase.LoadAssetAtPath<ObservationConfig>($"{ConfigDir}/ObservationConfig.asset");
            var traffic = AssetDatabase.LoadAssetAtPath<TrafficConfig>($"{ConfigDir}/TrafficConfig.asset");
            if (reward == null || obs == null || traffic == null)
            {
                Debug.LogError("[ElevatorRL] RewardConfig/ObservationConfig/TrafficConfig not found — " +
                                "run Tools > Elevator RL > Setup Scene once first to generate them.");
                return;
            }

            var go = GameObject.Find("ElevatorController");
            if (go == null) go = new GameObject("ElevatorController");

            var agent = go.GetComponent<ElevatorAttentionAgent>();
            if (agent == null) agent = go.AddComponent<ElevatorAttentionAgent>();
            agent.buildingConfig = buildingConfig;
            agent.rewardConfig = reward;
            agent.observationConfig = obs;
            agent.trafficConfig = traffic;

            var tempBuilding = new Building(buildingConfig, reward, obs, traffic, 1);
            int globalObsSize = tempBuilding.GlobalObservationSize();
            int carEntitySize = tempBuilding.CarEntitySize();

            var buffer = go.GetComponent<BufferSensorComponent>();
            if (buffer == null) buffer = go.AddComponent<BufferSensorComponent>();
            buffer.SensorName = "CarEntities";
            buffer.ObservableSize = carEntitySize;
            buffer.MaxNumObservables = buildingConfig.numElevators;

            int E = buildingConfig.numElevators;
            var branches = new int[E];
            for (int i = 0; i < E; i++) branches[i] = 6;

            var bp = go.GetComponent<BehaviorParameters>();
            if (bp == null) bp = go.AddComponent<BehaviorParameters>();
            bp.BehaviorName = "ElevatorAttention";
            bp.BehaviorType = BehaviorType.Default;
            bp.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(branches);
            bp.BrainParameters.VectorObservationSize = globalObsSize;

            EditorUtility.SetDirty(go);
            EditorSceneManager.MarkSceneDirty(go.scene);
            Debug.Log($"[ElevatorRL] E6 attention scene set up for {presetName} " +
                      $"({buildingConfig.numFloors}fl/{E}cars): ElevatorAttentionAgent, Behavior " +
                      $"'ElevatorAttention', globalObs={globalObsSize}, carEntity={carEntitySize} x{E} " +
                      $"(BufferSensor), action={E} branches x 6.");
        }
    }
}
