using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace ElevatorRL.Reports
{
    /// <summary>
    /// Filled mirrored-density shape for a violin plot — one strip of quads between a top edge and
    /// a bottom edge sampled at the same x positions. Semi-transparent fill; pair with a
    /// <see cref="UIPolyline"/> around the perimeter for a crisp outline (built by the caller from
    /// the same point arrays, since the outline needs the top edge forward + bottom edge reversed).
    /// </summary>
    [RequireComponent(typeof(CanvasRenderer))]
    public class UIViolinFill : MaskableGraphic
    {
        public List<Vector2> Top = new List<Vector2>();
        public List<Vector2> Bottom = new List<Vector2>();

        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();
            int n = Mathf.Min(Top.Count, Bottom.Count);
            if (n < 2) return;

            Color32 col = color;
            for (int i = 0; i < n - 1; i++)
            {
                int vi = vh.currentVertCount;
                AddVert(vh, Top[i], col);
                AddVert(vh, Top[i + 1], col);
                AddVert(vh, Bottom[i + 1], col);
                AddVert(vh, Bottom[i], col);
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

        public void SetShape(IList<Vector2> top, IList<Vector2> bottom)
        {
            Top.Clear(); Top.AddRange(top);
            Bottom.Clear(); Bottom.AddRange(bottom);
            SetVerticesDirty();
        }
    }
}
