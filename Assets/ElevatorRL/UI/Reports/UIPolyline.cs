using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace ElevatorRL.Reports
{
    /// <summary>
    /// Procedural polyline uGUI graphic — draws a connected line through <see cref="Points"/> (in
    /// the RectTransform's local rect space, pixel coordinates, not normalized). uGUI has no
    /// native line primitive (LineRenderer is a 3D/world-space component, unusable inside a
    /// Canvas), so this is the one piece of new chart infrastructure everything else — trajectory
    /// charts, violin outlines, ECDF-style curves — builds on.
    /// </summary>
    [RequireComponent(typeof(CanvasRenderer))]
    public class UIPolyline : MaskableGraphic
    {
        public List<Vector2> Points = new List<Vector2>();
        public float Thickness = 2f;

        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();
            if (Points == null || Points.Count < 2) return;

            Color32 col = color;
            for (int i = 0; i < Points.Count - 1; i++)
            {
                Vector2 a = Points[i], b = Points[i + 1];
                Vector2 dir = (b - a);
                if (dir.sqrMagnitude < 1e-8f) continue;
                dir.Normalize();
                Vector2 perp = new Vector2(-dir.y, dir.x) * (Thickness * 0.5f);

                int vi = vh.currentVertCount;
                AddVert(vh, a - perp, col);
                AddVert(vh, a + perp, col);
                AddVert(vh, b + perp, col);
                AddVert(vh, b - perp, col);
                vh.AddTriangle(vi, vi + 1, vi + 2);
                vh.AddTriangle(vi, vi + 2, vi + 3);
            }
        }

        static void AddVert(VertexHelper vh, Vector2 pos, Color32 col)
        {
            var v = UIVertex.simpleVert;
            v.color = col;
            v.position = pos;
            vh.AddVert(v);
        }

        public void SetPoints(IList<Vector2> pts)
        {
            Points.Clear();
            Points.AddRange(pts);
            SetVerticesDirty();
        }
    }
}
