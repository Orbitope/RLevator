using System;
using System.Collections.Generic;
using ContentKit;
using ElevatorRL.Stats;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace ElevatorRL.Reports
{
    /// <summary>
    /// Static chart-drawing helpers for the standard run report / overall roll-up scenes
    /// (STATS_VIZ_REPLAY_PLAN.md §6, ported from the validated HTML prototype). Every method
    /// places its output with explicit pixel geometry via <see cref="CKUI.TopLeft"/> — no
    /// layout-group timing to fight, matching ElevatorSandbox's convention. All Graphic-based
    /// children (UIPolyline/UIViolinFill) work in a rect whose LOCAL space has (0,0) at the
    /// visual top-left and +Y pointing DOWN (because TopLeft's pivot is (0,1)) — use
    /// <see cref="P"/> to convert a "pixels down from top" coordinate into that local space.
    /// </summary>
    public static class ReportWidgets
    {
        // ── palette (CKColor + two data-mark brights it doesn't define) ──────
        public static readonly Color Void = CKColor.Void, Surface = CKColor.Surface, Raised = CKColor.Raised,
            Border = CKColor.Border, TextBright = CKColor.TextBright, TextPrimary = CKColor.TextPrimary,
            TextSecondary = CKColor.TextSecondary, TextMuted = CKColor.TextMuted,
            Amber = CKColor.Amber, AmberBright = CKColor.AmberBright, Coral = CKColor.Coral,
            Steel = CKColor.Steel, SteelBright = CKColor.SteelBright, Sage = CKColor.Sage;
        public static readonly Color SageBright = CKColor.FromHex("#8FB07A");
        public static readonly Color CoralDim = CKColor.FromHex("#B6472C");

        static Vector2 P(float px, float py) => new Vector2(px, -py);

        static RectTransform Chart(Transform parent, float w, float h)
        {
            var rt = CKUI.EmptyRect(parent, "Chart");
            CKUI.TopLeft(rt, 0, 0, w, h);
            return rt;
        }

        static UIPolyline Line(Transform parent, Color c, float thickness = 1.5f)
        {
            var go = new GameObject("Line");
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            CKUI.Stretch(rt);
            var line = go.AddComponent<UIPolyline>();
            line.color = c; line.Thickness = thickness; line.raycastTarget = false;
            return line;
        }

        static void GridLine(Transform parent, float x1, float y1, float x2, float y2, Color c, float alpha = 0.5f)
        {
            var l = Line(parent, new Color(c.r, c.g, c.b, alpha), 1f);
            l.SetPoints(new[] { P(x1, y1), P(x2, y2) });
        }

        static TextMeshProUGUI Txt(Transform parent, string s, TMP_FontAsset font, float size, Color c,
            TextAlignmentOptions align, float x, float y, float w, float h)
        {
            var t = CKUI.Label(parent, "Txt", s, font, size, c, align);
            CKUI.TopLeft(t.rectTransform, x, y, w, h);
            return t;
        }

        // ── KPI tile ──────────────────────────────────────────────────────────
        public static void KpiTile(Transform parent, float x, float y, float w, float h, TMP_FontAsset mono,
            string label, float lookVal, float etaVal, string unit, int dec, bool higherBetter)
        {
            var tile = CKUI.Panel(parent, "Kpi_" + label, Surface);
            CKUI.TopLeft(tile, x, y, w, h);
            AddBorder(tile, Border);
            var bar = CKUI.Box(tile, "AccentBar", Amber);
            CKUI.TopLeft(bar.rectTransform, 0, 0, 2, h);

            Txt(tile, label.ToUpperInvariant(), mono, 9.5f, TextMuted, TextAlignmentOptions.TopLeft, 12, 8, w - 20, 12);
            Txt(tile, "LOOK", mono, 8f, TextMuted, TextAlignmentOptions.TopLeft, 12, 24, 50, 10);
            Txt(tile, "ETA", mono, 8f, TextMuted, TextAlignmentOptions.TopLeft, w * 0.55f, 24, 50, 10);
            Txt(tile, lookVal.ToString("F" + dec) + unit, mono, 15f, SteelBright, TextAlignmentOptions.TopLeft, 12, 34, w * 0.5f, 20);
            Txt(tile, etaVal.ToString("F" + dec) + unit, mono, 15f, SageBright, TextAlignmentOptions.TopLeft, w * 0.55f, 34, w * 0.45f, 20);

            float dpct = lookVal != 0 ? (etaVal - lookVal) / Mathf.Abs(lookVal) * 100f : 0f;
            bool same = Mathf.Abs(dpct) < 0.5f;
            bool better = higherBetter ? etaVal > lookVal : etaVal < lookVal;
            Color dc = same ? TextMuted : (better ? SageBright : Coral);
            string arrow = same ? "≈" : (etaVal > lookVal ? "▲" : "▼");
            CKUI.BorderTop(CKUI.EmptyRect(tile, "DeltaDiv").gameObject.transform, Border);
            var divRt = tile.Find("DeltaDiv").GetComponent<RectTransform>();
            CKUI.TopLeft(divRt, 12, h - 24, w - 24, 1);
            Txt(tile, $"{arrow} {(dpct >= 0 ? "+" : "")}{dpct:F1}% Δ", mono, 10f, dc, TextAlignmentOptions.TopLeft, 12, h - 18, w - 24, 14);
        }

        static void AddBorder(RectTransform rt, Color c)
        {
            CKUI.BorderTop(rt, c); CKUI.BorderBottom(rt, c); CKUI.BorderLeft(rt, c); CKUI.BorderRight(rt, c);
        }

        // ── trajectory mini chart (metric over WindowStats) ─────────────────
        public static void TrajectoryPanel(Transform parent, float x, float y, float w, float h, TMP_FontAsset mono,
            string label, string unit, float[] look, float[] eta)
        {
            var panel = CKUI.Panel(parent, "Traj_" + label, Surface);
            CKUI.TopLeft(panel, x, y, w, h);
            AddBorder(panel, Border);
            Txt(panel, label.ToUpperInvariant() + "  " + unit, mono, 9f, TextSecondary, TextAlignmentOptions.Top, 0, 8, w, 12);

            const float cm = 30, cx0 = 34, cx1w = 12, cy0 = 26, cy1 = 20;
            float cw = w - cx0 - cx1w, ch = h - cy0 - cy1;
            var chart = Chart(panel, cw, ch);
            CKUI.TopLeft(chart, cx0, cy0, cw, ch);

            int n = look.Length;
            float dmin = float.MaxValue, dmax = float.MinValue;
            foreach (var v in look) { dmin = Mathf.Min(dmin, v); dmax = Mathf.Max(dmax, v); }
            foreach (var v in eta) { dmin = Mathf.Min(dmin, v); dmax = Mathf.Max(dmax, v); }
            float pad = (dmax - dmin) * 0.2f; if (pad < 1e-4f) pad = Mathf.Max(0.1f, Mathf.Abs(dmax) * 0.1f);
            float lo = dmin - pad, hi = dmax + pad;

            float Xf(int i) => i / (float)(n - 1) * cw;
            float Yf(float v) => ch - (v - lo) / (hi - lo) * ch;

            GridLine(chart, 0, Yf(dmin), cw, Yf(dmin), Border, 0.4f);
            GridLine(chart, 0, Yf(dmax), cw, Yf(dmax), Border, 0.4f);
            Txt(panel, dmin.ToString("F0"), mono, 8f, TextMuted, TextAlignmentOptions.Right, 0, cy0 + Yf(dmin) - 5, cx0 - 4, 10);
            Txt(panel, dmax.ToString("F0"), mono, 8f, TextMuted, TextAlignmentOptions.Right, 0, cy0 + Yf(dmax) - 5, cx0 - 4, 10);

            foreach (var (arr, col) in new[] { (look, (Color)SteelBright), (eta, SageBright) })
            {
                var pts = new List<Vector2>(n);
                for (int i = 0; i < n; i++) pts.Add(P(Xf(i), Yf(arr[i])));
                Line(chart, col, 1.6f).SetPoints(pts);
                var dot = CKUI.Box(chart, "End", col);
                dot.rectTransform.anchorMin = dot.rectTransform.anchorMax = new Vector2(0, 1);
                dot.rectTransform.pivot = new Vector2(0.5f, 0.5f);
                dot.rectTransform.sizeDelta = new Vector2(5, 5);
                dot.rectTransform.anchoredPosition = P(Xf(n - 1), Yf(arr[n - 1]));
            }
        }

        // ── stacked violins (delivered-wait distribution per policy) ────────
        public static void ViolinPanel(Transform parent, float x, float y, float w, float h, TMP_FontAsset mono,
            (string name, Color col, List<WaitHistBin> hist)[] series, float xmax)
        {
            var panel = CKUI.Panel(parent, "Violin", Surface);
            CKUI.TopLeft(panel, x, y, w, h);
            AddBorder(panel, Border);

            const float padL = 46, padR = 16, padT = 8, padB = 20;
            float cw = w - padL - padR, rowH = (h - padT - padB) / series.Length;
            var chart = Chart(panel, cw, h - padT - padB);
            CKUI.TopLeft(chart, padL, padT, cw, h - padT - padB);

            foreach (var v in new float[] { 0, 15, 30, 45 })
            {
                if (v > xmax) continue;
                float xx = v / xmax * cw;
                GridLine(chart, xx, 0, xx, h - padT - padB, Border, 0.3f);
                Txt(panel, v.ToString("F0") + "s", mono, 8.5f, TextMuted, TextAlignmentOptions.Center, padL + xx - 14, h - padB + 4, 28, 12);
            }

            for (int si = 0; si < series.Length; si++)
            {
                var (name, col, hist) = series[si];
                float cy = rowH * si + rowH / 2f;
                int tot = 0, mx = 0; foreach (var b in hist) { tot += b.count; mx = Mathf.Max(mx, b.count); }
                float half = rowH / 2f - 8;

                var top = new List<Vector2>(hist.Count); var bot = new List<Vector2>(hist.Count);
                foreach (var b in hist)
                {
                    float xx = (b.start + b.end) * 0.5f / xmax * cw;
                    float t = mx > 0 ? b.count / (float)mx * half : 0f;
                    top.Add(P(xx, cy - t)); bot.Add(P(xx, cy + t));
                }
                var fillGo = new GameObject("Fill_" + name); fillGo.transform.SetParent(chart, false);
                var fillRt = fillGo.AddComponent<RectTransform>(); CKUI.Stretch(fillRt);
                var fill = fillGo.AddComponent<UIViolinFill>();
                fill.color = new Color(col.r, col.g, col.b, 0.26f); fill.raycastTarget = false;
                fill.SetShape(top, bot);

                GridLine(chart, 0, cy, cw, cy, col, 0.3f);

                // percentile ticks
                float Pct(float p) { int cum = 0; foreach (var b in hist) { cum += b.count; if (cum / (float)tot >= p) return (b.start + b.end) * 0.5f; } return xmax; }
                float p50 = Pct(0.5f), p95 = Pct(0.95f);
                var t50 = Line(chart, TextBright, 1.5f); t50.SetPoints(new[] { P(p50 / xmax * cw, cy - half), P(p50 / xmax * cw, cy + half) });
                var t95 = Line(chart, Coral, 2.2f); t95.SetPoints(new[] { P(p95 / xmax * cw, cy - half), P(p95 / xmax * cw, cy + half) });
                Txt(panel, "p95 " + p95.ToString("F0"), mono, 8.5f, Coral, TextAlignmentOptions.Center, padL + p95 / xmax * cw - 24, padT + cy - half - 12, 48, 10);
                Txt(panel, name, mono, 12f, col, TextAlignmentOptions.Right, 0, padT + cy - 7, padL - 12, 14);
            }
        }

        // ── reward decomposition (diverging horizontal bars) ─────────────────
        public static void RewardChart(Transform parent, float x, float y, float w, float h, TMP_FontAsset mono,
            EpisodeStats look, EpisodeStats eta, float maxAbs)
        {
            var panel = CKUI.Panel(parent, "Reward", Surface);
            CKUI.TopLeft(panel, x, y, w, h);
            AddBorder(panel, Border);

            string[] names = { "Delivered", "Toward", "Rejected", "Abandoned", "InElevator", "InQueue" };
            Func<EpisodeStats, float>[] get = {
                e => e.rwDelivered, e => e.rwToward, e => e.rwRejected, e => e.rwAbandoned, e => e.rwInElevator, e => e.rwInQueue };

            const float padL = 86, padR = 20, padT = 8, padB = 22;
            float cw = w - padL - padR, ch = h - padT - padB;
            var chart = Chart(panel, cw, ch);
            CKUI.TopLeft(chart, padL, padT, cw, ch);
            float zeroX = cw * 0.5f;
            GridLine(chart, zeroX, 0, zeroX, ch, TextMuted, 0.6f);
            foreach (var g in new float[] { -maxAbs * 0.5f, maxAbs * 0.5f })
            {
                float xx = zeroX + g / maxAbs * cw;
                GridLine(chart, xx, 0, xx, ch, Border, 0.4f);
                Txt(panel, (g / 1000f).ToString("F0") + "k", mono, 8.5f, TextMuted, TextAlignmentOptions.Center, padL + xx - 20, h - padB + 4, 40, 12);
            }

            float rowH = ch / names.Length;
            for (int i = 0; i < names.Length; i++)
            {
                float cy = rowH * (i + 0.5f);
                Txt(panel, names[i], mono, 10.5f, TextPrimary, TextAlignmentOptions.Right, 0, padT + cy - 7, padL - 10, 14);
                float lookV = get[i](look), etaV = get[i](eta);
                DivergingBar(chart, zeroX, cy - 7, cw, lookV, maxAbs, SteelBright);
                DivergingBar(chart, zeroX, cy + 2, cw, etaV, maxAbs, SageBright);
            }
        }

        static void DivergingBar(Transform chart, float zeroX, float y, float cw, float v, float maxAbs, Color col)
        {
            float xv = zeroX + v / maxAbs * cw * 0.5f;
            float x1 = Mathf.Min(zeroX, xv), width = Mathf.Abs(xv - zeroX);
            var bar = CKUI.Box(chart, "Bar", col);
            CKUI.TopLeft(bar.rectTransform, x1, y, Mathf.Max(width, 1f), 6);
        }

        // ── per-floor strip (small-multiple bars, one metric, both policies) ─
        public static void FloorStrip(Transform parent, float x, float y, float w, float h, TMP_FontAsset mono,
            string label, float[] look, float[] eta, float cap, string clipLabel)
        {
            Txt(parent, label, mono, 9.5f, TextSecondary, TextAlignmentOptions.TopLeft, x, y, w, 12);
            float chartY = y + 14, chartH = h - 14;
            var panel = CKUI.EmptyRect(parent, "FloorStrip_" + label);
            CKUI.TopLeft(panel, x, chartY, w, chartH);

            int F = look.Length;
            const float padL = 26, padB = 12;
            float cw = w - padL, ch = chartH - padB;
            var chart = Chart(panel, cw, ch);
            CKUI.TopLeft(chart, padL, 0, cw, ch);

            foreach (var t in new[] { cap * 0.5f, cap })
            {
                float yy = ch - t / cap * ch;
                GridLine(chart, 0, yy, cw, yy, Border, 0.45f);
                Txt(panel, t.ToString("F0"), mono, 8f, TextMuted, TextAlignmentOptions.Right, 0, yy - 5, padL - 4, 10);
            }

            float gw = cw / F, bw = Mathf.Min(4.2f, gw / 2f - 1f);
            for (int f = 0; f < F; f++)
            {
                float gx = gw * (f + 0.5f);
                DrawFloorBar(chart, gx - bw - 0.4f, look[f], cap, ch, SteelBright);
                DrawFloorBar(chart, gx + 0.4f, eta[f], cap, ch, SageBright);
                if (f == 0 && clipLabel != null)
                    Txt(panel, clipLabel, mono, 8.5f, AmberBright, TextAlignmentOptions.Left, padL + gx + 4, ch - ch, 60, 10);
                if (f % 3 == 0 || f == F - 1)
                    Txt(panel, f == 0 ? "G" : f.ToString(), mono, 8.5f, TextMuted, TextAlignmentOptions.Center, padL + gx - 10, ch + 2, 20, 10);
            }
        }

        static void DrawFloorBar(Transform chart, float bx, float v, float cap, float ch, Color col)
        {
            float vv = Mathf.Min(v, cap);
            float barH = vv / cap * ch;
            var bar = CKUI.Box(chart, "Bar", col);
            CKUI.TopLeft(bar.rectTransform, bx, ch - barH, 4.2f, Mathf.Max(barH, 0.6f));
        }

        // ── capacity bars (log-scale, one series, e.g. calibrated base intensity) ──
        public static void CapacityBars(Transform parent, float x, float y, float w, float h, TMP_FontAsset mono,
            string[] labels, float[] values, bool[] zoned)
        {
            var panel = CKUI.Panel(parent, "Capacity", Surface);
            CKUI.TopLeft(panel, x, y, w, h);
            AddBorder(panel, Border);

            const float padL = 40, padR = 12, padT = 10, padB = 26;
            float cw = w - padL - padR, ch = h - padT - padB;
            var chart = Chart(panel, cw, ch);
            CKUI.TopLeft(chart, padL, padT, cw, ch);

            float lo = Mathf.Log10(0.004f), hi = Mathf.Log10(2f);
            float Yf(float v) => ch - (Mathf.Log10(Mathf.Max(v, 1e-4f)) - lo) / (hi - lo) * ch;
            foreach (var g in new float[] { 0.01f, 0.1f, 1f })
            {
                float yy = Yf(g);
                GridLine(chart, 0, yy, cw, yy, Border, 0.5f);
                Txt(panel, g.ToString("0.##"), mono, 9f, TextMuted, TextAlignmentOptions.Right, 0, padT + yy - 5, padL - 6, 10);
            }

            int n = labels.Length; float gw = cw / n, bw = Mathf.Min(46f, gw * 0.5f);
            for (int i = 0; i < n; i++)
            {
                float cx = gw * (i + 0.5f);
                float yy = Yf(values[i]);
                var bar = CKUI.Box(chart, "Bar" + i, zoned[i] ? Amber : AmberBright);
                CKUI.TopLeft(bar.rectTransform, cx - bw / 2, yy, bw, ch - yy);
                Txt(panel, values[i].ToString(values[i] < 0.1f ? "F3" : "F2"), mono, 10.5f, TextBright,
                    TextAlignmentOptions.Center, padL + cx - 30, padT + yy - 16, 60, 14);
                Txt(panel, labels[i], mono, 11f, TextPrimary, TextAlignmentOptions.Center, padL + cx - 20, padT + ch + 4, 40, 14);
            }
        }

        // ── trellis cell (compact 2-series line, shared fixed y-axis) ──────
        public static void TrellisCell(Transform parent, float x, float y, float w, float h, Color border,
            float[] look, float[] eta, float ymax)
        {
            var panel = CKUI.Panel(parent, "Cell", Surface);
            CKUI.TopLeft(panel, x, y, w, h);
            AddBorder(panel, border);
            var chart = Chart(panel, w - 8, h - 8);
            CKUI.TopLeft(chart, 4, 4, w - 8, h - 8);
            float cw = w - 8, ch = h - 8;

            foreach (var g in new float[] { ch * 0.33f, ch * 0.66f })
                GridLine(chart, 0, g, cw, g, Border, 0.35f);

            int n = look.Length;
            float Xf(int i) => i / (float)(n - 1) * cw;
            float Yf(float v) => ch - Mathf.Clamp01(v / ymax) * ch;
            foreach (var (arr, col) in new[] { (look, (Color)SteelBright), (eta, SageBright) })
            {
                var pts = new List<Vector2>(n);
                for (int i = 0; i < n; i++) pts.Add(P(Xf(i), Yf(arr[i])));
                Line(chart, col, 1.4f).SetPoints(pts);
                for (int i = 0; i < n; i++)
                {
                    var dot = CKUI.Box(chart, "Dot", col);
                    dot.rectTransform.anchorMin = dot.rectTransform.anchorMax = new Vector2(0, 1);
                    dot.rectTransform.pivot = new Vector2(0.5f, 0.5f);
                    dot.rectTransform.sizeDelta = new Vector2(3.2f, 3.2f);
                    dot.rectTransform.anchoredPosition = P(Xf(i), Yf(arr[i]));
                }
            }
        }

        // ── numbers table (metric rows + inline sparkbar) ───────────────────
        public static float NumbersTable(Transform parent, float x, float y, float w, TMP_FontAsset mono,
            EpisodeStats look, EpisodeStats eta)
        {
            (string lab, float l, float e, int dec, string unit, bool pct)[] rows = {
                ("Delivered / hr", look.deliveredPerHour, eta.deliveredPerHour, 0, "/hr", false),
                ("Wait p50", look.waitP50, eta.waitP50, 1, "s", false),
                ("Wait p95", look.waitP95, eta.waitP95, 1, "s", false),
                ("Wait max", look.waitMax, eta.waitMax, 1, "s", false),
                ("Abandon rate", look.abandonRate * 100, eta.abandonRate * 100, 1, "%", true),
                ("Reject rate", look.rejectRate * 100, eta.rejectRate * 100, 1, "%", true),
                ("Fleet util", look.utilFleetMean, eta.utilFleetMean, 2, "", false),
                ("Total reward", look.rwTotal, eta.rwTotal, 0, "", false),
            };

            const float rowH = 26, headH = 20;
            var header = CKUI.Box(parent, "TblHead", Raised);
            CKUI.TopLeft(header.rectTransform, x, y, w, headH);
            string[] cols = { "Metric", "LOOK", "ETA", "LOOK vs ETA", "Δ ETA−LOOK" };
            float[] colX = { 0, w * 0.30f, w * 0.42f, w * 0.54f, w * 0.80f };
            for (int c = 0; c < cols.Length; c++)
                Txt(header.transform, cols[c].ToUpperInvariant(), mono, 8.5f, TextSecondary,
                    c == 0 ? TextAlignmentOptions.Left : TextAlignmentOptions.Left, x + colX[c] + 8, y + 4, w * 0.2f, 12);

            for (int i = 0; i < rows.Length; i++)
            {
                var (lab, l, e, dec, unit, isPct) = rows[i];
                float ry = y + headH + rowH * i;
                if (i % 2 == 1) { var bg = CKUI.Box(parent, "Zebra", new Color(1, 1, 1, 0.015f)); CKUI.TopLeft(bg.rectTransform, x, ry, w, rowH); }
                Txt(parent, lab, mono, 10.5f, TextPrimary, TextAlignmentOptions.Left, x + 8, ry + 6, w * 0.28f, 16);
                Txt(parent, l.ToString("F" + dec) + unit, mono, 10.5f, SteelBright, TextAlignmentOptions.Left, x + colX[1] + 8, ry + 6, w * 0.1f, 16);
                Txt(parent, e.ToString("F" + dec) + unit, mono, 10.5f, SageBright, TextAlignmentOptions.Left, x + colX[2] + 8, ry + 6, w * 0.1f, 16);

                float max = Mathf.Max(Mathf.Abs(l), Mathf.Abs(e), 1e-6f);
                float sparkW = w * 0.22f;
                var sparkArea = CKUI.EmptyRect(parent, "Spark");
                CKUI.TopLeft(sparkArea, x + colX[3] + 8, ry + 8, sparkW, rowH - 10);
                var lb = CKUI.Box(sparkArea, "L", SteelBright); CKUI.TopLeft(lb.rectTransform, 0, 0, Mathf.Max(2, Mathf.Abs(l) / max * sparkW), 4.5f);
                var eb = CKUI.Box(sparkArea, "E", SageBright); CKUI.TopLeft(eb.rectTransform, 0, 6, Mathf.Max(2, Mathf.Abs(e) / max * sparkW), 4.5f);

                float d = e - l; bool tiny = Mathf.Abs(d) < 0.05f;
                Txt(parent, (d >= 0 ? "+" : "") + d.ToString("F" + dec) + unit, mono, 10.5f, tiny ? TextMuted : TextSecondary,
                    TextAlignmentOptions.Left, x + colX[4] + 8, ry + 6, w * 0.18f, 16);
            }
            return headH + rowH * rows.Length;
        }
    }
}
