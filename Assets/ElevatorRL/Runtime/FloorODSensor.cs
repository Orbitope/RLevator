using UnityEngine;
using Unity.MLAgents.Sensors;

namespace ElevatorRL
{
    /// <summary>
    /// EXPERIMENT_PLAN.md E13d. Exposes the Building's hall origin×destination matrix as a
    /// (2 x F x F) visual observation — [channel = up/down] x [origin floor] x [destination floor] —
    /// so ML-Agents runs its convolutional visual encoder over a grid whose BOTH axes are ordered
    /// floors (unlike E13b's (1 x F x 8) grid, whose width axis was arbitrarily-ordered features).
    ///
    /// Set `vis_encode_type: resnet` in the yaml: F x F = 16x16 on rung M clears resnet's
    /// min-resolution of 15, which the 8-wide E13b grid could not (it was capped at match3).
    /// Pull-based like FloorGridSensor: reads the live Building each Write().
    /// </summary>
    public sealed class FloorODSensor : ISensor
    {
        readonly Building _building;
        readonly string _name;
        readonly ObservationSpec _spec;

        public FloorODSensor(Building building, string name = "FloorOD")
        {
            _building = building;
            _name = name;
            int F = building.cfg.numFloors;
            _spec = ObservationSpec.Visual(Building.FloorODChannels, F, F);
        }

        public ObservationSpec GetObservationSpec() => _spec;

        public int Write(ObservationWriter writer)
        {
            _building.WriteFloorOD(writer);
            int F = _building.cfg.numFloors;
            return Building.FloorODChannels * F * F;
        }

        public byte[] GetCompressedObservation() => System.Array.Empty<byte>();
        public CompressionSpec GetCompressionSpec() => CompressionSpec.Default();
        public string GetName() => _name;
        public void Update() { }
        public void Reset() { }
    }
}
