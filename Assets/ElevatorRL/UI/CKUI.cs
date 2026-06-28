using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace ElevatorRL
{
    /// <summary>
    /// Terse uGUI construction helpers used by ElevatorSandbox.
    /// </summary>
    internal static class CKUI
    {
        // ── Panels / boxes ──────────────────────────────────────────────────

        public static RectTransform Panel(Transform parent, string name, Color bg)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.color = bg;
            img.raycastTarget = false;
            return go.GetComponent<RectTransform>();
        }

        public static RectTransform EmptyRect(Transform parent, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            return go.AddComponent<RectTransform>();
        }

        public static Image Box(Transform parent, string name, Color c)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.color = c;
            img.raycastTarget = false;
            return img;
        }

        // ── Text ─────────────────────────────────────────────────────────────

        public static TextMeshProUGUI Label(Transform parent, string name, string text,
            TMP_FontAsset font, float size, Color color,
            TextAlignmentOptions align = TextAlignmentOptions.MidlineLeft,
            float charSpacing = 0f)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<TextMeshProUGUI>();
            t.text = text;
            if (font != null) t.font = font;
            t.fontSize = size;
            t.color = color;
            t.alignment = align;
            t.characterSpacing = charSpacing;
            t.raycastTarget = false;
            t.enableWordWrapping = false;
            t.overflowMode = TextOverflowModes.Overflow;
            return t;
        }

        // ── Anchoring helpers ────────────────────────────────────────────────

        /// Stretch to fill parent with optional insets.
        public static void Stretch(RectTransform rt, float l = 0, float r = 0, float t = 0, float b = 0)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(l, b);
            rt.offsetMax = new Vector2(-r, -t);
        }

        /// Anchor top-left, position in pixels from top-left corner.
        public static void TopLeft(RectTransform rt, float x, float y, float w, float h)
        {
            rt.anchorMin = new Vector2(0, 1);
            rt.anchorMax = new Vector2(0, 1);
            rt.pivot     = new Vector2(0, 1);
            rt.anchoredPosition = new Vector2(x, -y);
            rt.sizeDelta = new Vector2(w, h);
        }

        /// Anchor top edge, fill width, set height from top.
        public static void AnchorTop(RectTransform rt, float height)
        {
            rt.anchorMin = new Vector2(0, 1);
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(0, -height);
            rt.offsetMax = Vector2.zero;
        }

        /// Anchor bottom edge, fill width, set height from bottom.
        public static void AnchorBottom(RectTransform rt, float height)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = new Vector2(1, 0);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = new Vector2(0, height);
        }

        /// Fill between two vertical offsets from edges (topOffset from top, bottomOffset from bottom).
        public static void AnchorMiddle(RectTransform rt, float topOffset, float bottomOffset)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(0, bottomOffset);
            rt.offsetMax = new Vector2(0, -topOffset);
        }

        /// Anchor right side, fill height, set width.
        public static void AnchorRight(RectTransform rt, float width)
        {
            rt.anchorMin = new Vector2(1, 0);
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(-width, 0);
            rt.offsetMax = Vector2.zero;
        }

        /// Fill from left up to rightEdgeFromRight from right side.
        public static void AnchorLeft(RectTransform rt, float rightEdgeFromRight)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = new Vector2(1, 1);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = new Vector2(-rightEdgeFromRight, 0);
        }

        // ── 1px border lines ─────────────────────────────────────────────────

        public static Image BorderTop(Transform parent, Color c)
        {
            var img = Box(parent, "BorderTop", c);
            var rt = img.rectTransform;
            rt.anchorMin = new Vector2(0, 1);
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(0, -1);
            rt.offsetMax = Vector2.zero;
            return img;
        }

        public static Image BorderBottom(Transform parent, Color c)
        {
            var img = Box(parent, "BorderBot", c);
            var rt = img.rectTransform;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = new Vector2(1, 0);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = new Vector2(0, 1);
            return img;
        }

        public static Image BorderLeft(Transform parent, Color c)
        {
            var img = Box(parent, "BorderLeft", c);
            var rt = img.rectTransform;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = new Vector2(0, 1);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = new Vector2(1, 0);
            return img;
        }

        public static Image BorderRight(Transform parent, Color c)
        {
            var img = Box(parent, "BorderRight", c);
            var rt = img.rectTransform;
            rt.anchorMin = new Vector2(1, 0);
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(-1, 0);
            rt.offsetMax = Vector2.zero;
            return img;
        }

        // ── Button ────────────────────────────────────────────────────────────

        public static Button MakeButton(Transform parent, string name, string label,
            TMP_FontAsset font, float fontSize, Color bg, Color textColor,
            float minWidth = 0)
        {
            var rt = Panel(parent, name, bg);
            var img = rt.GetComponent<Image>();
            img.raycastTarget = true; // required for Button to receive clicks
            var btn = rt.gameObject.AddComponent<Button>();
            btn.targetGraphic = img;

            var txt = Label(rt, "Label", label, font, fontSize, textColor,
                TextAlignmentOptions.Midline);
            Stretch(txt.rectTransform);

            if (minWidth > 0)
            {
                var le = rt.gameObject.AddComponent<LayoutElement>();
                le.minWidth = minWidth;
            }

            return btn;
        }

        // ── Slider ────────────────────────────────────────────────────────────

        public static Slider MakeSlider(Transform parent, string name,
            float min, float max, float value, bool wholeNumbers,
            Color trackColor, Color fillColor)
        {
            var rt = EmptyRect(parent, name);
            var slider = rt.gameObject.AddComponent<Slider>();
            // Sliders intercept drag events — ensure the handle Image is raycasting
            // (background raycastTarget is already set to true below)
            slider.minValue = min;
            slider.maxValue = max;
            slider.value = value;
            slider.wholeNumbers = wholeNumbers;

            // background track
            var bg = Panel(rt, "Background", trackColor);
            Stretch(bg, 0, 0, 3, 3);
            bg.GetComponent<Image>().raycastTarget = true;

            // fill area
            var fillArea = EmptyRect(rt, "Fill Area");
            fillArea.anchorMin = new Vector2(0, 0.25f);
            fillArea.anchorMax = new Vector2(1, 0.75f);
            fillArea.offsetMin = new Vector2(5, 0);
            fillArea.offsetMax = new Vector2(-5, 0);

            var fill = Panel(fillArea, "Fill", fillColor);
            Stretch(fill.GetComponent<RectTransform>());

            // handle area
            var handleArea = EmptyRect(rt, "Handle Slide Area");
            Stretch(handleArea, 10, 10, 0, 0);

            var handle = Panel(handleArea, "Handle", fillColor);
            handle.sizeDelta = new Vector2(12, 12);
            handle.anchorMin = new Vector2(0, 0.5f);
            handle.anchorMax = new Vector2(0, 0.5f);
            handle.pivot = new Vector2(0.5f, 0.5f);
            var handleImg = handle.GetComponent<Image>();
            handleImg.raycastTarget = true; // handle must receive drag events

            slider.fillRect = fill.GetComponent<RectTransform>();
            slider.handleRect = handle;
            slider.targetGraphic = handleImg;

            return slider;
        }
    }
}
