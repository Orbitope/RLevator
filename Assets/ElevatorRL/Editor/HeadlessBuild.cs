using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Builds a standalone macOS player of Assets/Scenes/Training.unity for use as
    /// an ML-Agents training executable, launched headless (-batchmode -nographics)
    /// and in parallel via `mlagents-learn --env=... --num-envs=N`. This avoids
    /// training against the Editor (which always renders, and requires live
    /// Play-mode/MCP focus interaction).
    /// </summary>
    public static class HeadlessBuild
    {
        const string OutputPath = "Builds/HeadlessTrainer/RLevatorTrainer.app";
        const string MultiAgentOutputPath = "Builds/HeadlessTrainerMultiAgent/RLevatorTrainerMultiAgent.app";
        const string AttentionOutputPath = "Builds/HeadlessTrainerAttention/RLevatorTrainerAttention.app";

        [MenuItem("Tools/Elevator RL/Build Headless Trainer (macOS)")]
        public static void Build() => BuildScene("Assets/Scenes/Training.unity", OutputPath);

        // Per-rung persistent build paths — unlike the shared OutputPath above (which the next
        // rung's rebuild silently overwrites), these let every rung's build coexist on disk so a
        // whole batch of rungs can be pre-built in one Editor-responsive window, then trained
        // back-to-back with zero further Editor interaction (mlagents-learn only needs the .app,
        // not a live Editor). Point the agent at the matching preset + save the scene BEFORE
        // calling each of these — the build bakes whatever BuildingConfig is currently wired.
        [MenuItem("Tools/Elevator RL/Build Headless Trainer - S (macOS)")]
        public static void BuildS() => BuildScene("Assets/Scenes/Training.unity", "Builds/HeadlessTrainer_S/RLevatorTrainer.app");

        [MenuItem("Tools/Elevator RL/Build Headless Trainer - M (macOS)")]
        public static void BuildM() => BuildScene("Assets/Scenes/Training.unity", "Builds/HeadlessTrainer_M/RLevatorTrainer.app");

        [MenuItem("Tools/Elevator RL/Build Headless Trainer - L (macOS)")]
        public static void BuildL() => BuildScene("Assets/Scenes/Training.unity", "Builds/HeadlessTrainer_L/RLevatorTrainer.app");

        [MenuItem("Tools/Elevator RL/Build Headless Trainer - Z (macOS)")]
        public static void BuildZ() => BuildScene("Assets/Scenes/Training.unity", "Builds/HeadlessTrainer_Z/RLevatorTrainer.app");

        [MenuItem("Tools/Elevator RL/Build Headless Trainer - H (macOS)")]
        public static void BuildH() => BuildScene("Assets/Scenes/Training.unity", "Builds/HeadlessTrainer_H/RLevatorTrainer.app");

        [MenuItem("Tools/Elevator RL/E6 Multi-Agent/Build Headless Trainer (macOS)")]
        public static void BuildMultiAgent() => BuildScene("Assets/Scenes/TrainingMultiAgent.unity", MultiAgentOutputPath);

        [MenuItem("Tools/Elevator RL/E6 Attention/Build Headless Trainer (macOS)")]
        public static void BuildAttention() => BuildScene("Assets/Scenes/TrainingAttention.unity", AttentionOutputPath);

        static void BuildScene(string scenePath, string outputPath)
        {
            var report = BuildPipeline.BuildPlayer(
                new[] { scenePath },
                outputPath,
                BuildTarget.StandaloneOSX,
                BuildOptions.None);

            var summary = report.summary;
            UnityEngine.Debug.Log($"[ElevatorRL] Headless trainer build ({scenePath}): {summary.result}, " +
                                  $"{summary.totalErrors} errors, {summary.totalWarnings} warnings, " +
                                  $"output={summary.outputPath}");

            if (summary.result == BuildResult.Succeeded)
                MarkBackgroundOnly(outputPath);
        }

        // mlagents-learn --num-envs=N launches N copies of this build as plain child
        // processes; without this, each one registers as a normal GUI app and gets its
        // own Dock icon even under -batchmode -nographics. LSBackgroundOnly suppresses
        // that (no Dock icon, no menu bar, no Cmd-Tab entry) for every future rebuild.
        static void MarkBackgroundOnly(string appPath)
        {
            string plistPath = Path.Combine(appPath, "Contents", "Info.plist");
            if (!File.Exists(plistPath))
            {
                UnityEngine.Debug.LogWarning($"[ElevatorRL] No Info.plist found at {plistPath}, skipping LSBackgroundOnly patch.");
                return;
            }

            string plist = File.ReadAllText(plistPath);
            const string key = "<key>LSBackgroundOnly</key>";
            if (plist.Contains(key)) return;

            const string versionKey = "<key>CFBundleVersion</key>";
            int idx = plist.IndexOf(versionKey);
            if (idx < 0)
            {
                UnityEngine.Debug.LogWarning("[ElevatorRL] Could not find insertion point in Info.plist for LSBackgroundOnly.");
                return;
            }
            int lineEnd = plist.IndexOf('\n', plist.IndexOf('\n', idx) + 1) + 1;
            plist = plist.Insert(lineEnd, "    <key>LSBackgroundOnly</key>\n    <true />\n");
            File.WriteAllText(plistPath, plist);
            UnityEngine.Debug.Log("[ElevatorRL] Patched Info.plist with LSBackgroundOnly (no Dock icon for headless instances).");
        }
    }
}
