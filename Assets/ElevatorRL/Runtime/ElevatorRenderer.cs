using System.Collections.Generic;
using UnityEngine;

namespace ElevatorRL
{
    /// <summary>
    /// Minimal, optional read-only visualization built from runtime primitives, themed
    /// in ContentKit colors (amber = cars, steel = structure). It mirrors a Building
    /// each FixedUpdate and never mutates env state. Swap for the full ContentKit scene
    /// / CKHUD when you film. Out-of-service cars are greyed and parked translucent.
    /// </summary>
    public sealed class ElevatorRenderer : MonoBehaviour
    {
        [Header("Layout (world units)")]
        public float floorHeight = 1.4f;
        public float shaftSpacing = 1.2f;
        public float carWidth = 0.9f;

        // ContentKit palette
        static readonly Color Void = new Color(0.067f, 0.063f, 0.035f);
        static readonly Color Steel = new Color(0.42f, 0.478f, 0.553f);
        static readonly Color Amber = new Color(0.769f, 0.604f, 0.235f);
        static readonly Color AmberBright = new Color(0.91f, 0.753f, 0.408f);
        static readonly Color Coral = new Color(1f, 0.369f, 0.227f);
        static readonly Color OutOfService = new Color(0.24f, 0.23f, 0.2f);

        Building _b;
        Transform[] _cars;
        Renderer[] _carR;
        Transform[] _carFill; // load bar that scales with occupancy
        Material _matMain;
        bool _built;

        void Build(Building b)
        {
            _b = b;
            int F = b.cfg.numFloors, E = b.cfg.numElevators;
            Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            _matMain = new Material(shader);

            // floor slabs
            for (int f = 0; f < F; f++)
            {
                var slab = GameObject.CreatePrimitive(PrimitiveType.Cube);
                slab.transform.SetParent(transform, false);
                slab.transform.localScale = new Vector3(E * shaftSpacing + carWidth, 0.04f, 1f);
                slab.transform.localPosition = new Vector3((E - 1) * shaftSpacing * 0.5f, f * floorHeight, 0f);
                Paint(slab, Steel * 0.7f);
            }

            _cars = new Transform[E];
            _carR = new Renderer[E];
            _carFill = new Transform[E];
            for (int i = 0; i < E; i++)
            {
                // shaft guide
                var shaft = GameObject.CreatePrimitive(PrimitiveType.Cube);
                shaft.transform.SetParent(transform, false);
                shaft.transform.localScale = new Vector3(carWidth + 0.1f, (F - 1) * floorHeight + floorHeight, 0.05f);
                shaft.transform.localPosition = new Vector3(i * shaftSpacing, (F - 1) * floorHeight * 0.5f, 0.4f);
                Paint(shaft, Steel * 0.25f);

                // car
                var car = GameObject.CreatePrimitive(PrimitiveType.Cube);
                car.transform.SetParent(transform, false);
                car.transform.localScale = new Vector3(carWidth, floorHeight * 0.82f, 0.6f);
                _cars[i] = car.transform;
                _carR[i] = car.GetComponent<Renderer>();
                Paint(car, Amber);

                // load bar
                var fill = GameObject.CreatePrimitive(PrimitiveType.Cube);
                fill.transform.SetParent(car.transform, false);
                fill.transform.localScale = new Vector3(0.18f, 0.9f, 1.05f);
                fill.transform.localPosition = new Vector3(-0.45f, 0f, 0f);
                _carFill[i] = fill.transform;
                Paint(fill, AmberBright);
            }

            _built = true;
        }

        public void Mirror(Building b)
        {
            if (!_built) Build(b);
            if (_b != b) _b = b;

            for (int i = 0; i < _cars.Length; i++)
            {
                var c = b.cars[i];
                _cars[i].localPosition = new Vector3(i * shaftSpacing, c.position * floorHeight, 0f);

                Color col;
                if (!c.inService) col = OutOfService;
                else if (c.state == CarState.DoorsOpening || c.state == CarState.Dwelling || c.state == CarState.DoorsClosing) col = AmberBright;
                else col = Amber;
                _carR[i].material.color = col;
                _cars[i].gameObject.SetActive(true);

                float load = c.inService ? c.Load : 0f;
                _carFill[i].localScale = new Vector3(0.18f, Mathf.Max(0.02f, 0.9f * load), 1.05f);
                _carFill[i].localPosition = new Vector3(-0.45f, -0.45f + 0.45f * Mathf.Max(0.02f, 0.9f * load), 0f);
                _carFill[i].gameObject.SetActive(c.inService);
            }
        }

        void Paint(GameObject go, Color c)
        {
            var r = go.GetComponent<Renderer>();
            r.material = new Material(_matMain) { color = c };
        }
    }
}
