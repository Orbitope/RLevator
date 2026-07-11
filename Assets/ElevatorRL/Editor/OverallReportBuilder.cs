using System.Collections.Generic;
using System.IO;
using System.Linq;
using ContentKit;
using ElevatorRL.Reports;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using UnityEditor.SceneManagement;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Builds the overall/E1 roll-up scene from an EvalHarness sweep summary.csv
    /// (RunSweepCalibrated output) — the calibrated-capacity chart and the small-multiples
    /// trellis (abandon rate vs. load, every rung × pattern, LOOK vs. ETA).
    /// </summary>
    public static class OverallReportBuilder
    {
        const string MonoPath = "Assets/ElevatorRL/UI/Fonts/JetBrainsMono SDF.asset";
        const string DisplayPath = "Assets/ElevatorRL/UI/Fonts/Rajdhani SDF.asset";
        const string ReportScenePath = "Assets/Scenes/OverallReport.unity";
        const float W = 1000f, Margin = 20f, ContentW = W - Margin * 2f;
        static readonly string[] RungOrder = { "S", "M", "L", "Z", "H" };

        [MenuItem("Tools/Elevator RL/Build Overall Report Scene (pick sweep summary.csv)")]
        static void BuildInteractive()
        {
            string runsRoot = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Runs");
            string csvPath = EditorUtility.OpenFilePanel("Select sweep summary.csv", runsRoot, "csv");
            if (string.IsNullOrEmpty(csvPath)) return;
            Build(csvPath);
        }

        // Non-interactive entry point (no file-picker dialog) — points at this session's E1
        // calibrated sweep. Edit the path to target a different sweep.
        [MenuItem("Tools/Elevator RL/Build Overall Report Scene (demo: E1 calibrated)")]
        static void BuildDemo()
        {
            string runsRoot = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Runs");
            Build(Path.Combine(runsRoot, "20260709-225210-E1-calibrated", "summary.csv"));
        }

        static void Build(string csvPath)
        {
            var rows = ReportData.ReadSummary(csvPath);
            var mono = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(MonoPath);
            var display = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(DisplayPath);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var canvasGO = new GameObject("Canvas");
            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGO.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(W, 1200);
            scaler.matchWidthOrHeight = 0f;
            canvasGO.AddComponent<GraphicRaycaster>();

            var esGO = new GameObject("EventSystem");
            esGO.AddComponent<UnityEngine.EventSystems.EventSystem>();
            esGO.AddComponent<UnityEngine.InputSystem.UI.InputSystemUIInputModule>();

            var rootGO = new GameObject("OverallReport");
            rootGO.transform.SetParent(canvasGO.transform, false);
            var root = rootGO.AddComponent<RectTransform>();
            root.anchorMin = new Vector2(0.5f, 1); root.anchorMax = new Vector2(0.5f, 1);
            root.pivot = new Vector2(0.5f, 1); root.sizeDelta = new Vector2(W, 100);
            var bg = rootGO.AddComponent<Image>(); bg.color = ReportWidgets.Void; bg.raycastTarget = false;

            var rungs = RungOrder.Where(r => rows.Any(row => row.preset == r)).ToArray();
            var patterns = rows.Select(r => r.pattern).Distinct().ToArray();

            float y = 0f;
            y += Header(root, mono, display, rows.Count);
            y += 20;
            y += SectionLabel(root, mono, display, y, "SCALE · CALIBRATED DEMAND", "EFFECTIVE CAPACITY BY RUNG");
            y += CapacitySection(root, mono, y, rows, rungs, patterns.Length > 0 ? patterns[0] : "");
            y += 30;
            y += SectionLabel(root, mono, display, y, "SMALL MULTIPLES · THE WHOLE SWEEP", "EVERY RUNG × PATTERN AT A GLANCE");
            y += TrellisSection(root, mono, y, rows, rungs, patterns);
            y += 40;

            root.sizeDelta = new Vector2(W, y);
            scaler.referenceResolution = new Vector2(W, y);

            Directory.CreateDirectory(Path.GetDirectoryName(ReportScenePath));
            EditorSceneManager.SaveScene(scene, ReportScenePath);
            Debug.Log($"[OverallReportBuilder] Built overall report from {csvPath} ({rows.Count} rows, " +
                      $"{rungs.Length} rungs x {patterns.Length} patterns). Height={y:F0}px. Saved to {ReportScenePath}");
            Selection.activeGameObject = rootGO;
        }

        static float Header(RectTransform root, TMP_FontAsset mono, TMP_FontAsset display, int rowCount)
        {
            var kicker = CKUI.Label(root, "Kicker", "OVERALL ROLL-UP · E1 BASELINE CHARACTERIZATION", mono, 9.5f,
                ReportWidgets.TextMuted, TextAlignmentOptions.TopLeft, 4f);
            CKUI.TopLeft(kicker.rectTransform, Margin, 10, 600, 14);
            var title = CKUI.Label(root, "Title", "ELEVATOR", display, 30f, ReportWidgets.TextBright,
                TextAlignmentOptions.TopLeft, 2f);
            title.fontStyle = FontStyles.Bold;
            CKUI.TopLeft(title.rectTransform, Margin, 26, 400, 44);
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

        static float CapacitySection(RectTransform root, TMP_FontAsset mono, float y, List<SummaryRow> rows, string[] rungs, string pattern)
        {
            const float h = 240f;
            var values = new float[rungs.Length];
            var zoned = new bool[rungs.Length];
            for (int i = 0; i < rungs.Length; i++)
            {
                var match = rows.FirstOrDefault(r => r.preset == rungs[i] && r.pattern == pattern);
                values[i] = match.calibratedBase;
                zoned[i] = rungs[i] == "Z" || rungs[i] == "H";
            }
            ReportWidgets.CapacityBars(root, Margin, y, ContentW, h, mono, rungs, values, zoned);
            return h;
        }

        static float TrellisSection(RectTransform root, TMP_FontAsset mono, float y, List<SummaryRow> rows, string[] rungs, string[] patterns)
        {
            const float gap = 6f, rowLabelW = 60f;
            float cellW = (ContentW - rowLabelW - gap * rungs.Length) / rungs.Length;
            float cellH = 88f;

            for (int c = 0; c < rungs.Length; c++)
            {
                var lab = CKUI.Label(root, "Col" + c, rungs[c], mono, 13f, ReportWidgets.TextPrimary, TextAlignmentOptions.Center);
                lab.fontStyle = FontStyles.Bold;
                CKUI.TopLeft(lab.rectTransform, Margin + rowLabelW + c * (cellW + gap), y, cellW, 16);
            }
            float gridY = y + 20;

            var mult = rows.Select(r => r.intensityMultiplier).Distinct().OrderBy(v => v).ToArray();
            for (int p = 0; p < patterns.Length; p++)
            {
                float ry = gridY + p * (cellH + gap);
                var plab = CKUI.Label(root, "Row" + p, patterns[p], mono, 9.5f, ReportWidgets.TextSecondary, TextAlignmentOptions.MidlineRight);
                CKUI.TopLeft(plab.rectTransform, Margin, ry, rowLabelW - 8, cellH);

                for (int c = 0; c < rungs.Length; c++)
                {
                    var look = new float[mult.Length]; var eta = new float[mult.Length];
                    for (int m = 0; m < mult.Length; m++)
                    {
                        var lr = rows.FirstOrDefault(r => r.preset == rungs[c] && r.pattern == patterns[p] && r.policy == "LOOK" && Mathf.Approximately(r.intensityMultiplier, mult[m]));
                        var er = rows.FirstOrDefault(r => r.preset == rungs[c] && r.pattern == patterns[p] && r.policy == "ETA" && Mathf.Approximately(r.intensityMultiplier, mult[m]));
                        look[m] = lr.abandonRateMean; eta[m] = er.abandonRateMean;
                    }
                    bool zoned = rungs[c] == "Z" || rungs[c] == "H";
                    ReportWidgets.TrellisCell(root, Margin + rowLabelW + c * (cellW + gap), ry, cellW, cellH,
                        zoned ? ReportWidgets.CoralDim : ReportWidgets.Border, look, eta, 0.35f);
                }
            }
            return 20f + patterns.Length * (cellH + gap);
        }
    }
}
