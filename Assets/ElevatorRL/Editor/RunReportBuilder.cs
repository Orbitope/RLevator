using System.Collections.Generic;
using System.IO;
using ContentKit;
using ElevatorRL.Reports;
using ElevatorRL.Stats;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using UnityEditor.SceneManagement;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Builds the standard per-run report scene (STATS_VIZ_REPLAY_PLAN.md §6 / the validated HTML
    /// prototype) from a pair of EvalHarness run folders — one per policy. Reads the canonical CSV
    /// tables via ReportData and lays out the same sections as the HTML: KPI strip, per-metric
    /// trajectory small multiples, wait-distribution violins, reward decomposition, per-floor
    /// strips, and a numbers table with inline sparkbars.
    /// </summary>
    public static class RunReportBuilder
    {
        const string MonoPath = "Assets/ElevatorRL/UI/Fonts/JetBrainsMono SDF.asset";
        const string DisplayPath = "Assets/ElevatorRL/UI/Fonts/Rajdhani SDF.asset";
        const string ReportScenePath = "Assets/Scenes/RunReport.unity";
        const float W = 1000f, Margin = 20f, ContentW = W - Margin * 2f;

        [MenuItem("Tools/Elevator RL/Build Run Report Scene (pick LOOK + ETA run folders)")]
        static void BuildInteractive()
        {
            string runsRoot = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Runs");
            string lookDir = EditorUtility.OpenFolderPanel("Select LOOK run folder", runsRoot, "");
            if (string.IsNullOrEmpty(lookDir)) return;
            string etaDir = EditorUtility.OpenFolderPanel("Select ETA run folder", runsRoot, "");
            if (string.IsNullOrEmpty(etaDir)) return;
            Build(lookDir, etaDir);
        }

        // Non-interactive entry point (no folder-picker dialog) for scripted/automated builds —
        // points at this session's L / Up-Peak / seed 1 run pair. Edit the folder names to target
        // a different run.
        [MenuItem("Tools/Elevator RL/Build Run Report Scene (demo: L UpPeak)")]
        static void BuildDemo()
        {
            string runsRoot = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Runs");
            Build(Path.Combine(runsRoot, "20260710-165614-LOOK-L-UpPeak-i0.4-s1"),
                  Path.Combine(runsRoot, "20260710-165615-ETA-L-UpPeak-i0.4-s1"));
        }

        static void Build(string lookDir, string etaDir)
        {
            var look = ReportData.ReadEpisode(Path.Combine(lookDir, "episode.csv"));
            var eta = ReportData.ReadEpisode(Path.Combine(etaDir, "episode.csv"));
            var lookFloors = ReportData.ReadFloors(Path.Combine(lookDir, "floor_stats.csv"));
            var etaFloors = ReportData.ReadFloors(Path.Combine(etaDir, "floor_stats.csv"));
            var lookWin = ReportData.ReadWindows(Path.Combine(lookDir, "window_stats.csv"));
            var etaWin = ReportData.ReadWindows(Path.Combine(etaDir, "window_stats.csv"));

            List<WaitHistBin> lookHist = null, etaHist = null;
            string lookHistPath = Path.Combine(lookDir, "wait_hist.csv"), etaHistPath = Path.Combine(etaDir, "wait_hist.csv");
            if (File.Exists(lookHistPath) && File.Exists(etaHistPath))
            {
                lookHist = ReportData.ReadWaitHist(lookHistPath);
                etaHist = ReportData.ReadWaitHist(etaHistPath);
            }

            var mono = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(MonoPath);
            var display = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(DisplayPath);
            if (mono == null) Debug.LogWarning("[RunReportBuilder] Mono font not found at " + MonoPath);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var canvasGO = new GameObject("Canvas");
            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGO.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(W, 1400);
            scaler.matchWidthOrHeight = 0f;
            canvasGO.AddComponent<GraphicRaycaster>();

            var esGO = new GameObject("EventSystem");
            esGO.AddComponent<UnityEngine.EventSystems.EventSystem>();
            esGO.AddComponent<UnityEngine.InputSystem.UI.InputSystemUIInputModule>();

            var rootGO = new GameObject("RunReport");
            rootGO.transform.SetParent(canvasGO.transform, false);
            var root = rootGO.AddComponent<RectTransform>();
            root.anchorMin = new Vector2(0.5f, 1); root.anchorMax = new Vector2(0.5f, 1);
            root.pivot = new Vector2(0.5f, 1); root.sizeDelta = new Vector2(W, 100); // height fixed up below
            var bg = rootGO.AddComponent<Image>(); bg.color = ReportWidgets.Void; bg.raycastTarget = false;

            float y = 0f;
            y += Header(root, mono, display, look);
            y += 20;
            y += SectionLabel(root, mono, display, y, "SUMMARY · EPISODESTATS", "HEADLINE METRICS");
            y += KpiStrip(root, mono, y, look, eta);
            y += 30;
            y += SectionLabel(root, mono, display, y, "TRAJECTORY · WINDOWSTATS", "OVER THE EPISODE");
            y += TrajectoryStrip(root, mono, y, lookWin, etaWin);
            y += 30;
            if (lookHist != null)
            {
                y += SectionLabel(root, mono, display, y, "DISTRIBUTION · WAIT HISTOGRAM", "WAIT-TIME DISTRIBUTION");
                y += ViolinSection(root, mono, y, lookHist, etaHist);
                y += 30;
            }
            y += SectionLabel(root, mono, display, y, "WHY · REWARD TERMS", "REWARD DECOMPOSITION");
            y += RewardSection(root, mono, y, look, eta);
            y += 30;
            y += SectionLabel(root, mono, display, y, "WHERE · FLOORSTATS", "PER-FLOOR BREAKDOWN");
            y += FloorSection(root, mono, y, lookFloors, etaFloors);
            y += 30;
            y += SectionLabel(root, mono, display, y, "RAW · ALL TABLES", "NUMBERS");
            y += ReportWidgets.NumbersTable(root, Margin, y, ContentW, mono, look, eta);
            y += 40;

            root.sizeDelta = new Vector2(W, y);
            scaler.referenceResolution = new Vector2(W, y);

            Directory.CreateDirectory(Path.GetDirectoryName(ReportScenePath));
            EditorSceneManager.SaveScene(scene, ReportScenePath);
            Debug.Log($"[RunReportBuilder] Built report: LOOK={look.id.policy}@{lookDir}, ETA={eta.id.policy}@{etaDir}. " +
                      $"Height={y:F0}px. Saved to {ReportScenePath}");
            Selection.activeGameObject = rootGO;
        }

        static float Header(RectTransform root, TMP_FontAsset mono, TMP_FontAsset display, EpisodeStats look)
        {
            var kicker = CKUI.Label(root, "Kicker", "STANDARD RUN REPORT", mono, 9.5f, ReportWidgets.TextMuted, TextAlignmentOptions.TopLeft, 5f);
            CKUI.TopLeft(kicker.rectTransform, Margin, 10, 400, 14);
            var title = CKUI.Label(root, "Title", $"{look.id.buildingPreset} · {look.id.pattern}", display, 28f, ReportWidgets.TextBright,
                TextAlignmentOptions.TopLeft, 2f);
            title.fontStyle = FontStyles.Bold | FontStyles.UpperCase;
            CKUI.TopLeft(title.rectTransform, Margin, 26, 500, 40);

            string[] chips = {
                $"{look.id.buildingPreset}", $"intensity {look.id.intensity:F3}", $"seed {look.id.seed}",
                $"{look.id.simSeconds / 60f:F0} min sim" };
            float cx = W - Margin;
            for (int i = chips.Length - 1; i >= 0; i--)
            {
                float cw = chips[i].Length * 6.2f + 16;
                cx -= cw;
                var chip = CKUI.Panel(root, "Chip" + i, ReportWidgets.Surface);
                CKUI.TopLeft(chip, cx, 20, cw, 22);
                var t = CKUI.Label(chip, "T", chips[i], mono, 9.5f, ReportWidgets.TextSecondary, TextAlignmentOptions.Center);
                CKUI.Stretch(t.rectTransform);
                cx -= 6;
            }

            var div = CKUI.Box(root, "HeaderDiv", ReportWidgets.Border);
            CKUI.TopLeft(div.rectTransform, Margin, 78, ContentW, 1);
            return 90f;
        }

        static float SectionLabel(RectTransform root, TMP_FontAsset mono, TMP_FontAsset display, float y, string eyebrow, string title)
        {
            var eb = CKUI.Label(root, "Eyebrow", eyebrow, mono, 9.5f, ReportWidgets.TextMuted, TextAlignmentOptions.TopLeft, 2.5f);
            CKUI.TopLeft(eb.rectTransform, Margin, y, ContentW, 12);
            var h2 = CKUI.Label(root, "H2", title, display, 18f, ReportWidgets.TextBright, TextAlignmentOptions.TopLeft, 1f);
            h2.fontStyle = FontStyles.Bold | FontStyles.UpperCase;
            CKUI.TopLeft(h2.rectTransform, Margin, y + 14, ContentW, 24);
            return 44f;
        }

        static float KpiStrip(RectTransform root, TMP_FontAsset mono, float y, EpisodeStats look, EpisodeStats eta)
        {
            (string lab, float l, float e, string unit, int dec, bool higher)[] rows = {
                ("Throughput", look.deliveredPerHour, eta.deliveredPerHour, "/hr", 0, true),
                ("Wait p50", look.waitP50, eta.waitP50, "s", 1, false),
                ("Wait p95", look.waitP95, eta.waitP95, "s", 1, false),
                ("Wait max", look.waitMax, eta.waitMax, "s", 1, false),
                ("Abandon", look.abandonRate * 100, eta.abandonRate * 100, "%", 1, false),
                ("Reject", look.rejectRate * 100, eta.rejectRate * 100, "%", 1, false),
                ("Fleet util", look.utilFleetMean, eta.utilFleetMean, "", 2, true),
                ("Reward", look.rwTotal, eta.rwTotal, "", 0, true),
            };
            const int cols = 4; float gap = 8f, tw = (ContentW - gap * (cols - 1)) / cols, th = 92f;
            for (int i = 0; i < rows.Length; i++)
            {
                int col = i % cols, row = i / cols;
                float x = Margin + col * (tw + gap), ty = y + row * (th + gap);
                ReportWidgets.KpiTile(root, x, ty, tw, th, mono, rows[i].lab, rows[i].l, rows[i].e, rows[i].unit, rows[i].dec, rows[i].higher);
            }
            int rowsCount = Mathf.CeilToInt(rows.Length / (float)cols);
            return rowsCount * (th + gap);
        }

        static float TrajectoryStrip(RectTransform root, TMP_FontAsset mono, float y, List<WindowStats> look, List<WindowStats> eta)
        {
            int n = Mathf.Min(look.Count, eta.Count);
            float[] rateL = new float[n], rateE = new float[n], wpL = new float[n], wpE = new float[n],
                abL = new float[n], abE = new float[n], utilL = new float[n], utilE = new float[n];
            for (int i = 0; i < n; i++)
            {
                rateL[i] = look[i].deliveredRate; rateE[i] = eta[i].deliveredRate;
                wpL[i] = look[i].waitP95; wpE[i] = eta[i].waitP95;
                abL[i] = look[i].abandoned; abE[i] = eta[i].abandoned;
                utilL[i] = look[i].fleetUtilMean; utilE[i] = eta[i].fleetUtilMean;
            }
            const int cols = 4; float gap = 8f, tw = (ContentW - gap * (cols - 1)) / cols, th = 128f;
            ReportWidgets.TrajectoryPanel(root, Margin + 0 * (tw + gap), y, tw, th, mono, "Throughput", "/hr", rateL, rateE);
            ReportWidgets.TrajectoryPanel(root, Margin + 1 * (tw + gap), y, tw, th, mono, "Wait p95", "s", wpL, wpE);
            ReportWidgets.TrajectoryPanel(root, Margin + 2 * (tw + gap), y, tw, th, mono, "Abandoned", "/win", abL, abE);
            ReportWidgets.TrajectoryPanel(root, Margin + 3 * (tw + gap), y, tw, th, mono, "Fleet util", "", utilL, utilE);
            return th;
        }

        static float ViolinSection(RectTransform root, TMP_FontAsset mono, float y, List<WaitHistBin> look, List<WaitHistBin> eta)
        {
            const float h = 140f;
            float xmax = look.Count > 0 ? look[look.Count - 1].end : 45f;
            ReportWidgets.ViolinPanel(root, Margin, y, ContentW, h, mono,
                new (string, Color, List<WaitHistBin>)[] { ("LOOK", ReportWidgets.SteelBright, look), ("ETA", ReportWidgets.SageBright, eta) }, xmax);
            return h;
        }

        static float RewardSection(RectTransform root, TMP_FontAsset mono, float y, EpisodeStats look, EpisodeStats eta)
        {
            const float h = 210f;
            float maxAbs = 1000f;
            foreach (var v in new[] { look.rwDelivered, look.rwToward, look.rwRejected, look.rwAbandoned, look.rwInElevator, look.rwInQueue,
                                       eta.rwDelivered, eta.rwToward, eta.rwRejected, eta.rwAbandoned, eta.rwInElevator, eta.rwInQueue })
                maxAbs = Mathf.Max(maxAbs, Mathf.Abs(v));
            maxAbs *= 1.1f;
            ReportWidgets.RewardChart(root, Margin, y, ContentW, h, mono, look, eta, maxAbs);
            return h;
        }

        static float FloorSection(RectTransform root, TMP_FontAsset mono, float y, List<FloorStats> look, List<FloorStats> eta)
        {
            int F = Mathf.Min(look.Count, eta.Count);
            float[] dL = new float[F], dE = new float[F], aL = new float[F], aE = new float[F], wL = new float[F], wE = new float[F];
            float capD = 10, capA = 5, capW = 45;
            for (int f = 0; f < F; f++)
            {
                dL[f] = look[f].delivered; dE[f] = eta[f].delivered; capD = Mathf.Max(capD, Mathf.Max(dL[f], dE[f]));
                aL[f] = look[f].abandoned; aE[f] = eta[f].abandoned; capA = Mathf.Max(capA, Mathf.Max(aL[f], aE[f]));
                wL[f] = look[f].waitP95; wE[f] = eta[f].waitP95;
            }
            capD *= 1.08f; capA *= 1.08f;
            string clipD = dL[0] > capD || dE[0] > capD ? $"{dL[0]:F0}/{dE[0]:F0} ▲" : null;
            string clipA = aL[0] > capA || aE[0] > capA ? $"{aL[0]:F0}/{aE[0]:F0} ▲" : null;

            const float stripH = 78f;
            var panel = CKUI.Panel(root, "FloorPanel", ReportWidgets.Surface);
            CKUI.TopLeft(panel, Margin, y, ContentW, stripH * 3 + 12);
            ReportWidgets.FloorStrip(panel, 10, 6, ContentW - 20, stripH, mono, "DELIVERED / FLOOR", dL, dE, capD, clipD);
            ReportWidgets.FloorStrip(panel, 10, 6 + stripH, ContentW - 20, stripH, mono, "ABANDONED / FLOOR", aL, aE, capA, clipA);
            ReportWidgets.FloorStrip(panel, 10, 6 + stripH * 2, ContentW - 20, stripH, mono, "WAIT P95 / FLOOR (s)", wL, wE, capW, null);
            return stripH * 3 + 12;
        }
    }
}
