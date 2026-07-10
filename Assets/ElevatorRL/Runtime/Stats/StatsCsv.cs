using System.Collections.Generic;
using System.IO;

namespace ElevatorRL.Stats
{
    /// <summary>Writes stat records to CSV files (header + rows). Engine-light (System.IO only).</summary>
    public static class StatsCsv
    {
        public static void Write(string path, string header, IEnumerable<string> rows)
        {
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            using (var w = new StreamWriter(path, false))
            {
                w.WriteLine(header);
                foreach (var r in rows) w.WriteLine(r);
            }
        }
    }
}
