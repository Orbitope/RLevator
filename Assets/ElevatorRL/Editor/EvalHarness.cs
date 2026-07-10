using System;
using System.Collections.Generic;
using System.IO;
using ElevatorRL.Stats;
using UnityEngine;
using UnityEditor;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Headless evaluation runner (seed of the M1 sweep). For now: a single deterministic LOOK
    /// episode that drives a StatsCollector and writes the three CSV tables — the end-to-end
    /// smoke test for the M0 instrumentation.
    /// </summary>
    public static class EvalHarness
    {
        [MenuItem("Tools/Elevator RL/Run Eval (LOOK smoke test)")]
        static void RunLookSmokeTest()
        {
            var summary = RunSingle(
                policy: "LOOK",
                preset: "S-smoke",
                floors: 8, cars: 3, capacity: 8,
                pattern: TrafficPattern.UpPeak, intensity: 1f,
                seed: 1, totalSeconds: 3600f, warmupSeconds: 300f, bucketSeconds: 300f);

            Debug.Log("[Eval] " + summary);
        }

        /// <summary>Runs one LOOK episode headlessly, writes CSVs, returns a one-line summary.</summary>
        public static string RunSingle(string policy, string preset, int floors, int cars, int capacity,
            TrafficPattern pattern, float intensity, int seed,
            float totalSeconds, float warmupSeconds, float bucketSeconds)
        {
            // self-contained configs (don't touch saved assets)
            var cfg = ScriptableObject.CreateInstance<BuildingConfig>();
            cfg.numFloors = floors; cfg.numElevators = cars; cfg.capacity = capacity;
            cfg.randomizeActive = false; cfg.minActiveElevators = 1; cfg.serviceChangeProbability = 0f;

            var reward  = ScriptableObject.CreateInstance<RewardConfig>();
            var obs     = ScriptableObject.CreateInstance<ObservationConfig>();
            var traffic = ScriptableObject.CreateInstance<TrafficConfig>();
            traffic.useDayCycle = false; traffic.defaultPattern = pattern; traffic.intensity = intensity;

            var b = new Building(cfg, reward, obs, traffic, seed);
            b.Reset();

            var col = new StatsCollector(b, warmupSeconds, bucketSeconds);

            const float dt = 0.05f;
            float decInterval = cfg.decisionInterval;
            float clock = 0f;
            int guard = 0, guardMax = (int)(totalSeconds / dt) + 1000;

            while (b.simTime < totalSeconds && guard++ < guardMax)
            {
                b.Tick(dt);
                clock += dt;
                if (clock >= decInterval)
                {
                    clock -= decInterval;
                    var act = ElevatorHeuristics.CollectiveLook(b);
                    for (int i = 0; i < cfg.numElevators; i++) b.ApplyAction(i, act[i]);
                    b.CollectReward();     // accumulates reward decomposition
                    col.Sample();          // occupancy + window rolling
                }
            }

            var id = new RunId
            {
                runId = $"{DateTime.Now:yyyyMMdd-HHmmss}-{policy}-{preset}",
                gitSha = "", policy = policy, modelPath = "",
                buildingPreset = preset, configHash = "", trafficPreset = "inline",
                pattern = pattern.ToString(), intensity = intensity, seed = seed,
            };

            var episode = col.Finish(id);
            var floorsRows = col.BuildFloorStats(id);
            var windowRows = col.BuildWindowStats(id);
            col.Dispose();

            // write CSVs to <project>/Runs/<runId>/
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string runDir = Path.Combine(projectRoot, "Runs", id.runId);

            StatsCsv.Write(Path.Combine(runDir, "episode.csv"),
                EpisodeStats.Header, new List<string> { episode.ToCsv() });

            var floorLines = new List<string>(floorsRows.Count);
            foreach (var fs in floorsRows) floorLines.Add(fs.ToCsv());
            StatsCsv.Write(Path.Combine(runDir, "floor_stats.csv"), FloorStats.Header, floorLines);

            var winLines = new List<string>(windowRows.Count);
            foreach (var ws in windowRows) winLines.Add(ws.ToCsv());
            StatsCsv.Write(Path.Combine(runDir, "window_stats.csv"), WindowStats.Header, winLines);

            return $"{id.runId}: delivered={episode.delivered} " +
                   $"waitMean={episode.waitMean:0.0}s p95={episode.waitP95:0.0}s max={episode.waitMax:0.0}s " +
                   $"abandoned={episode.abandoned} rejected={episode.rejected} " +
                   $"util={episode.utilFleetMean:0.00} rwTotal={episode.rwTotal:0} " +
                   $"[{floorsRows.Count} floor rows, {windowRows.Count} window rows] → {runDir}";
        }
    }
}
