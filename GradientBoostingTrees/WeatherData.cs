using System.Globalization;

namespace GradientBoostingTrees
{
    public class WeatherData
    {
        public DateTime Timestamp { get; set; }
        public double Temperature { get; set; }
    }

    public class DataProcessor
    {
        public List<WeatherData> LoadWeatherData(string filePath)
        {
            var records = new List<WeatherData>();
            bool dataStarted = false;

            using (var reader = new StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    // Check for the actual data start point
                    if (!dataStarted)
                    {
                        if (line.StartsWith("timestamp"))
                        {
                            dataStarted = true; // Data starts from the next line
                        }
                        continue;
                    }

                    var values = line.Split(',');
                    if (values.Length >= 2)
                    {
                        if (DateTime.TryParseExact(values[0], "yyyyMMddTHHmm", CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime timestamp) &&
                            double.TryParse(values[1], out double temperature))
                        {
                            records.Add(new WeatherData { Timestamp = timestamp, Temperature = temperature });
                        }
                    }
                }
            }
            return records;
        }

        public (double[][] Features, double[] Targets) PreprocessData(List<WeatherData> data, DateTime excludeDate)
        {
            var filteredData = data.Where(d => d.Timestamp.Date != excludeDate.Date).ToList();
            double[][] features = filteredData.Select(d => new double[] { d.Timestamp.DayOfYear, d.Timestamp.Hour }).ToArray();
            double[] targets = filteredData.Select(d => d.Temperature).ToArray();
            return (features, targets);
        }
    }
}