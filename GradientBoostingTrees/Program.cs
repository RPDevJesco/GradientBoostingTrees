namespace GradientBoostingTrees
{
    public class Program
    {
        public static void Main(string[] args)
        {
            WeatherExample();
        }

        private static void WeatherExample()
        {
            // margin of error of 2-3°C
            string filePath = "weatheryearcomparison.csv";

            var processor = new DataProcessor();
            var weatherData = processor.LoadWeatherData(filePath);
            var (features, targets) = processor.PreprocessData(weatherData, DateTime.Today);

            // Initialize loss function and gradient boosting model
            var lossFunction = new MeanSquaredError();
            var gradientBoosting = new GradientBoosting(lossFunction, 0.05);

            // Train the model with the preprocessed data
            gradientBoosting.Fit(features, targets, 200, 4);

            // Predict today's weather
            double[] todayFeatures = new double[] { DateTime.Today.DayOfYear, DateTime.Now.Hour };
            double prediction = gradientBoosting.Predict(todayFeatures);

            // Output the prediction result
            Console.WriteLine($"Predicted temperature for today: {prediction}");
        }
    }
}