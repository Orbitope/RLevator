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

        [MenuItem("Tools/Elevator RL/Build Headless Trainer (macOS)")]
        public static void Build()
        {
            var report = BuildPipeline.BuildPlayer(
                new[] { "Assets/Scenes/Training.unity" },
                OutputPath,
                BuildTarget.StandaloneOSX,
                BuildOptions.None);

            var summary = report.summary;
            UnityEngine.Debug.Log($"[ElevatorRL] Headless trainer build: {summary.result}, " +
                                  $"{summary.totalErrors} errors, {summary.totalWarnings} warnings, " +
                                  $"output={summary.outputPath}");

            if (summary.result == BuildResult.Succeeded)
                MarkBackgroundOnly(OutputPath);
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
