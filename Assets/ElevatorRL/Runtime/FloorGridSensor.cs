using UnityEngine;
using Unity.MLAgents.Sensors;

namespace ElevatorRL
{
    /// <summary>
    /// EXPERIMENT_PLAN.md E13b. A custom pull-based <see cref="ISensor"/> that exposes the Building's
    /// per-floor local state as a single-channel (1 x F x <see cref="Building.FloorGridFeatures"/>)
    /// visual/grid observation, so ML-Agents automatically applies its convolutional visual encoder
    /// (set vis_encode_type: match3 in the yaml -- min resolution 5, satisfied by F>=5 floors and
    /// 8 features) and convolves over the FLOOR axis. This is the native, ONNX-exportable analog of
    /// the 2024 Traffic-Pattern-Aware paper's Conv1d-over-floors, avoiding the custom-torch/ONNX
    /// fragility this project hit before. Pull-based: reads the live Building each Write() rather than
    /// being pushed data, so the owning agent just needs to hand it the Building reference.
    /// </summary>
    public sealed class FloorGridSensor : ISensor
    {
        readonly Building _building;
        readonly string _name;
        readonly ObservationSpec _spec;

        public FloorGridSensor(Building building, string name = "FloorGrid")
        {
            _building = building;
            _name = name;
            // channels=1, height=numFloors, width=per-floor feature count
            _spec = ObservationSpec.Visual(1, building.cfg.numFloors, Building.FloorGridFeatures);
        }

        public ObservationSpec GetObservationSpec() => _spec;

        public int Write(ObservationWriter writer)
        {
            _building.WriteFloorGrid(writer);
            return _building.cfg.numFloors * Building.FloorGridFeatures;
        }

        public byte[] GetCompressedObservation() => System.Array.Empty<byte>();
        public CompressionSpec GetCompressionSpec() => CompressionSpec.Default();
        public string GetName() => _name;
        public void Update() { }
        public void Reset() { }
    }
}
